import os
import sys
import time
import json
import collections

sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using computing device:", device)

DATASETS = ["movie", "book", "music"]
DATA_DIR_BASE = os.path.abspath("backend/data")
OUT_DIR = os.path.abspath("results")
FIG_DIR = os.path.abspath("docs/figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# -------------------------------------------------------------
# MODEL DEFINITIONS
# -------------------------------------------------------------
class MostPopularRecommender:
    def __init__(self):
        self.item_counts = collections.defaultdict(int)
        self.max_count = 1.0

    def fit(self, train_data):
        for u, i, r in train_data:
            if r == 1:
                self.item_counts[i] += 1
        self.max_count = max(self.item_counts.values()) if self.item_counts else 1.0

    def predict(self, items):
        return np.array([self.item_counts.get(i, 0) / self.max_count for i in items])

class ItemKNNRecommender:
    def __init__(self, k=20):
        self.k = k
        self.user_pos = collections.defaultdict(set)
        self.item_pos = collections.defaultdict(set)

    def fit(self, train_data):
        for u, i, r in train_data:
            if r == 1:
                self.user_pos[u].add(i)
                self.item_pos[i].add(u)

    def predict_one(self, u, i):
        u_liked = self.user_pos.get(u, set())
        if not u_liked:
            return 0.5
        i_users = self.item_pos.get(i, set())
        if not i_users:
            return 0.5
        sims = []
        for liked_item in u_liked:
            liked_users = self.item_pos.get(liked_item, set())
            denom = np.sqrt(len(i_users) * len(liked_users))
            if denom > 0:
                sims.append(len(i_users & liked_users) / denom)
        if not sims:
            return 0.5
        return float(np.mean(sorted(sims, reverse=True)[:self.k]))

    def predict(self, users, items):
        return np.array([self.predict_one(u, i) for u, i in zip(users, items)])

class MatrixFactorization(nn.Module):
    def __init__(self, n_user, n_item, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_user, dim)
        self.item_emb = nn.Embedding(n_item, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.item_bias = nn.Embedding(n_item, 1)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        u = self.user_emb(users)
        v = self.item_emb(items)
        dot = (u * v).sum(dim=-1, keepdim=True)
        logits = dot + self.user_bias(users) + self.item_bias(items)
        return torch.sigmoid(logits.squeeze(-1))

class CKAN(nn.Module):
    def __init__(self, n_entity, n_relation, dim=64, n_layer=1, agg="concat"):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_layer = n_layer
        self.agg = agg

        self.entity_emb = nn.Embedding(n_entity, dim)
        self.relation_emb = nn.Embedding(n_relation, dim)
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.ReLU(),
            nn.Linear(dim, dim, bias=False), nn.ReLU(),
            nn.Linear(dim, 1, bias=False), nn.Sigmoid()
        )
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        for m in self.attention:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        alpha = self.attention(torch.cat((h_emb, r_emb), dim=-1)).squeeze(-1)
        alpha = F.softmax(alpha, dim=-1)
        return torch.mul(alpha.unsqueeze(-1), t_emb).sum(dim=1)

    def get_user_embeddings(self, user_triple):
        user_embs = [self.entity_emb(user_triple[0][0]).mean(dim=1)]
        for l in range(self.n_layer):
            h = self.entity_emb(user_triple[0][l])
            r = self.relation_emb(user_triple[1][l])
            t = self.entity_emb(user_triple[2][l])
            user_embs.append(self._knowledge_attention(h, r, t))
        e_u = user_embs[0]
        if self.agg == "concat":
            for i in range(1, len(user_embs)):
                e_u = torch.cat((user_embs[i], e_u), dim=-1)
        elif self.agg == "sum":
            for i in range(1, len(user_embs)):
                e_u = e_u + user_embs[i]
        return e_u

    def get_item_embeddings(self, items, item_triple):
        item_embs = [self.entity_emb(items)]
        for l in range(self.n_layer):
            h = self.entity_emb(item_triple[0][l])
            r = self.relation_emb(item_triple[1][l])
            t = self.entity_emb(item_triple[2][l])
            item_embs.append(self._knowledge_attention(h, r, t))
        e_v = item_embs[0]
        if self.agg == "concat":
            for i in range(1, len(item_embs)):
                e_v = torch.cat((item_embs[i], e_v), dim=-1)
        elif self.agg == "sum":
            for i in range(1, len(item_embs)):
                e_v = e_v + item_embs[i]
        return e_v

    def forward(self, items, user_triple, item_triple):
        e_u = self.get_user_embeddings(user_triple)
        e_v = self.get_item_embeddings(items, item_triple)
        return torch.sigmoid((e_v * e_u).sum(dim=-1))

# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------
def to_triple_tensor(objs, triple_set, n_layer, dev):
    h, r, t = [], [], []
    for i in range(n_layer):
        h.append(torch.LongTensor([triple_set[o][i][0] for o in objs]).to(dev))
        r.append(torch.LongTensor([triple_set[o][i][1] for o in objs]).to(dev))
        t.append(torch.LongTensor([triple_set[o][i][2] for o in objs]).to(dev))
    return [h, r, t]

def evaluate_predictions(labels, scores):
    auc = roc_auc_score(labels, scores)
    preds = [1 if s >= 0.5 else 0 for s in scores]
    f1 = f1_score(labels, preds)
    return auc, f1

# -------------------------------------------------------------
# MAIN BENCHMARK ENGINE ACROSS 3 DATASETS
# -------------------------------------------------------------
def run_benchmark():
    all_results = []
    print("="*70)
    print("BẮT ĐẦU CHẠY THỰC NGHIỆM ĐA MIỀN TRÊN 3 TẬP DỮ LIỆU CHUẨN")
    print("="*70)

    for ds_name in DATASETS:
        ds_dir = os.path.join(DATA_DIR_BASE, ds_name)
        r_file = os.path.join(ds_dir, "ratings_final.npy")
        k_file = os.path.join(ds_dir, "kg_final.npy")

        if not os.path.exists(r_file) or not os.path.exists(k_file):
            print(f"Skipping {ds_name}: files not found.")
            continue

        print(f"\n>>> [DATASET: {ds_name.upper()}] <<<")
        rating_np = np.load(r_file)
        kg_np = np.load(k_file)

        n_user = int(np.max(rating_np[:, 0])) + 1
        n_item = int(np.max(rating_np[:, 1])) + 1
        n_entity = int(max(np.max(kg_np[:, 0]), np.max(kg_np[:, 2]), np.max(rating_np[:, 1]))) + 1
        n_relation = int(np.max(kg_np[:, 1])) + 1
        print(f"  Users: {n_user:,} | Items: {n_item:,} | Ratings: {len(rating_np):,} | Entities: {n_entity:,} | Relations: {n_relation:,} | Triples: {len(kg_np):,}")

        # Split 6:2:2
        np.random.seed(42)
        n_rows = len(rating_np)
        idx = np.random.permutation(n_rows)
        train_data = rating_np[idx[:int(0.6 * n_rows)]]
        test_data = rating_np[idx[int(0.8 * n_rows):]]

        # Build KG dict & Ripple sets
        kg_dict = defaultdict(list)
        for h, r, t in kg_np:
            kg_dict[int(h)].append((int(t), int(r)))

        n_layer = 1
        itss = 32 if ds_name == "music" else 64
        utss = 16 if ds_name == "book" else 32

        # Item ripple
        item_triple_set = defaultdict(list)
        for item in range(n_item):
            for l in range(n_layer):
                neighbors = kg_dict.get(item, [])
                if len(neighbors) == 0:
                    h, r, t = [item]*itss, [0]*itss, [item]*itss
                else:
                    indices = np.random.choice(len(neighbors), size=itss, replace=(len(neighbors) < itss))
                    h = [item for _ in range(itss)]
                    r = [neighbors[i][1] for i in indices]
                    t = [neighbors[i][0] for i in indices]
                item_triple_set[item].append((h, r, t))

        # User ripple
        user_history = defaultdict(list)
        for u, i, r in train_data:
            if r == 1:
                user_history[u].append(i)

        user_triple_set = defaultdict(list)
        for u in range(n_user):
            liked = user_history.get(u, [0])
            entities = liked
            for l in range(n_layer):
                h, r, t = [], [], []
                for ent in entities:
                    for tail, rel in kg_dict.get(ent, []):
                        h.append(ent)
                        r.append(rel)
                        t.append(tail)
                if len(h) == 0:
                    user_triple_set[u].append(([liked[0]]*utss, [0]*utss, [liked[0]]*utss))
                else:
                    indices = np.random.choice(len(h), size=utss, replace=(len(h) < utss))
                    user_triple_set[u].append(([h[i] for i in indices], [r[i] for i in indices], [t[i] for i in indices]))
                    entities = [t[i] for i in indices]

        # 1. MostPopular
        pop = MostPopularRecommender()
        pop.fit(train_data)
        pop_sc = pop.predict(test_data[:, 1])
        pop_auc, pop_f1 = evaluate_predictions(test_data[:, 2], pop_sc)
        print(f"  [1/4] MostPop  : AUC = {pop_auc:.4f} | F1 = {pop_f1:.4f}")

        # 2. Item-KNN (Sample 3000 test points for speed)
        knn = ItemKNNRecommender(k=20)
        knn.fit(train_data)
        sub_test = test_data[:min(3000, len(test_data))]
        knn_sc = knn.predict(sub_test[:, 0], sub_test[:, 1])
        knn_auc, knn_f1 = evaluate_predictions(sub_test[:, 2], knn_sc)
        print(f"  [2/4] Item-KNN : AUC = {knn_auc:.4f} | F1 = {knn_f1:.4f}")

        # 3. Matrix Factorization
        mf = MatrixFactorization(n_user, n_item, dim=64).to(device)
        opt_mf = torch.optim.Adam(mf.parameters(), lr=0.002, weight_decay=1e-5)
        crit = nn.BCELoss()
        bs = 2048

        for ep in range(5):
            mf.train()
            perm = np.random.permutation(len(train_data))
            for s in range(0, len(train_data), bs):
                b = train_data[perm[s:s+bs]]
                u = torch.LongTensor(b[:, 0]).to(device)
                i = torch.LongTensor(b[:, 1]).to(device)
                lbl = torch.FloatTensor(b[:, 2]).to(device)
                opt_mf.zero_grad()
                l = crit(mf(u, i), lbl)
                l.backward()
                opt_mf.step()

        mf.eval()
        with torch.no_grad():
            mf_sc = []
            for s in range(0, len(test_data), bs):
                b = test_data[s:s+bs]
                sc = mf(torch.LongTensor(b[:, 0]).to(device), torch.LongTensor(b[:, 1]).to(device)).cpu().numpy()
                mf_sc.extend(sc)
        mf_auc, mf_f1 = evaluate_predictions(test_data[:, 2], mf_sc)
        print(f"  [3/4] Biased MF: AUC = {mf_auc:.4f} | F1 = {mf_f1:.4f}")

        # 4. CKAN
        ckan = CKAN(n_entity, n_relation, dim=64, n_layer=n_layer, agg="concat").to(device)
        opt_ckan = torch.optim.Adam(ckan.parameters(), lr=0.002, weight_decay=1e-5)

        for ep in range(5):
            ckan.train()
            perm = np.random.permutation(len(train_data))
            for s in range(0, len(train_data), bs):
                b = train_data[perm[s:s+bs]]
                i = torch.LongTensor(b[:, 1]).to(device)
                lbl = torch.FloatTensor(b[:, 2]).to(device)
                u_tr = to_triple_tensor(b[:, 0], user_triple_set, n_layer, device)
                i_tr = to_triple_tensor(b[:, 1], item_triple_set, n_layer, device)
                opt_ckan.zero_grad()
                l = crit(ckan(i, u_tr, i_tr), lbl)
                l.backward()
                opt_ckan.step()

        ckan.eval()
        with torch.no_grad():
            ckan_sc = []
            for s in range(0, len(test_data), bs):
                b = test_data[s:s+bs]
                i = torch.LongTensor(b[:, 1]).to(device)
                u_tr = to_triple_tensor(b[:, 0], user_triple_set, n_layer, device)
                i_tr = to_triple_tensor(b[:, 1], item_triple_set, n_layer, device)
                sc = ckan(i, u_tr, i_tr).cpu().numpy()
                ckan_sc.extend(sc)
        ckan_auc, ckan_f1 = evaluate_predictions(test_data[:, 2], ckan_sc)
        print(f"  [4/4] CKAN (KG): AUC = {ckan_auc:.4f} | F1 = {ckan_f1:.4f}")

        # 5. Sparsity test (10% data)
        sub_size = int(len(train_data) * 0.1)
        sub_tr = train_data[:sub_size]
        mf_sp = MatrixFactorization(n_user, n_item, dim=64).to(device)
        opt_sp = torch.optim.Adam(mf_sp.parameters(), lr=0.002, weight_decay=1e-5)
        for _ in range(4):
            mf_sp.train()
            for s in range(0, len(sub_tr), bs):
                b = sub_tr[s:s+bs]
                opt_sp.zero_grad()
                l = crit(mf_sp(torch.LongTensor(b[:, 0]).to(device), torch.LongTensor(b[:, 1]).to(device)), torch.FloatTensor(b[:, 2]).to(device))
                l.backward()
                opt_sp.step()
        with torch.no_grad():
            mf_sp_sc = []
            for s in range(0, len(test_data), bs):
                b = test_data[s:s+bs]
                mf_sp_sc.extend(mf_sp(torch.LongTensor(b[:, 0]).to(device), torch.LongTensor(b[:, 1]).to(device)).cpu().numpy())
        mf_sparse_auc, _ = evaluate_predictions(test_data[:, 2], mf_sp_sc)

        # CKAN Sparsity (10% data)
        ckan_sp = CKAN(n_entity, n_relation, dim=64, n_layer=n_layer, agg="concat").to(device)
        opt_ck_sp = torch.optim.Adam(ckan_sp.parameters(), lr=0.002, weight_decay=1e-5)
        for _ in range(4):
            ckan_sp.train()
            for s in range(0, len(sub_tr), bs):
                b = sub_tr[s:s+bs]
                u_tr = to_triple_tensor(b[:, 0], user_triple_set, n_layer, device)
                i_tr = to_triple_tensor(b[:, 1], item_triple_set, n_layer, device)
                opt_ck_sp.zero_grad()
                l = crit(ckan_sp(torch.LongTensor(b[:, 1]).to(device), u_tr, i_tr), torch.FloatTensor(b[:, 2]).to(device))
                l.backward()
                opt_ck_sp.step()
        with torch.no_grad():
            ckan_sp_sc = []
            for s in range(0, len(test_data), bs):
                b = test_data[s:s+bs]
                u_tr = to_triple_tensor(b[:, 0], user_triple_set, n_layer, device)
                i_tr = to_triple_tensor(b[:, 1], item_triple_set, n_layer, device)
                ckan_sp_sc.extend(ckan_sp(torch.LongTensor(b[:, 1]).to(device), u_tr, i_tr).cpu().numpy())
        ckan_sparse_auc, _ = evaluate_predictions(test_data[:, 2], ckan_sp_sc)

        print(f"  [Sparsity 10%] MF AUC: {mf_sparse_auc:.4f} | CKAN AUC: {ckan_sparse_auc:.4f} (Chênh lệch: +{(ckan_sparse_auc - mf_sparse_auc)*100:.1f}%)")

        all_results.append({
            "Dataset": ds_name.capitalize(),
            "MostPop_AUC": pop_auc,
            "ItemKNN_AUC": knn_auc,
            "MF_AUC": mf_auc,
            "CKAN_AUC": ckan_auc,
            "MF_F1": mf_f1,
            "CKAN_F1": ckan_f1,
            "MF_Sparse10_AUC": mf_sparse_auc,
            "CKAN_Sparse10_AUC": ckan_sparse_auc,
        })

    # Save summary dataframe
    df = pd.DataFrame(all_results)
    out_csv = os.path.join(OUT_DIR, "tri_dataset_benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    print("\n" + "="*70)
    print("BẢNG TỔNG KẾT THỰC NGHIỆM ĐA MIỀN TRÊN 3 TẬP DỮ LIỆU:")
    print("="*70)
    print(df.to_string(index=False))

    # Generate Visualization Figure 17
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Warm-start CTR Comparison across 3 datasets
    x = np.arange(len(df))
    width = 0.20
    ax1.bar(x - 1.5*width, df["MostPop_AUC"], width, label="MostPopular", color="#9CA3AF")
    ax1.bar(x - 0.5*width, df["ItemKNN_AUC"], width, label="Item-KNN", color="#60A5FA")
    ax1.bar(x + 0.5*width, df["MF_AUC"], width, label="Matrix Factorization", color="#3B82F6")
    ax1.bar(x + 1.5*width, df["CKAN_AUC"], width, label="CKAN (Proposed)", color="#D97706")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Dataset"], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Test ROC-AUC", fontsize=11, fontweight="bold")
    ax1.set_title("So Sánh ROC-AUC Trên 3 Miền Dữ Liệu (Warm-Start)", fontsize=11.5, fontweight="bold")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", linestyle=":", alpha=0.7)

    # 2. Sparsity 10% Comparison across 3 datasets
    width2 = 0.35
    ax2.bar(x - width2/2, df["MF_Sparse10_AUC"], width2, label="MF (10% Dữ Liệu)", color="#93C5FD", edgecolor="#3B82F6")
    ax2.bar(x + width2/2, df["CKAN_Sparse10_AUC"], width2, label="CKAN (10% Dữ Liệu)", color="#F59E0B", edgecolor="#D97706")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Dataset"], fontsize=11, fontweight="bold")
    ax2.set_ylabel("Test ROC-AUC", fontsize=11, fontweight="bold")
    ax2.set_title("Khả Năng Chống Chịu Độ Thưa Thớt (10% Data)", fontsize=11.5, fontweight="bold", color="#991B1B")
    ax2.set_ylim(0.4, 1.05)
    ax2.legend(loc="lower right")
    ax2.grid(axis="y", linestyle=":", alpha=0.7)

    # Add text labels on bars
    for i in range(len(df)):
        diff = (df["CKAN_Sparse10_AUC"][i] - df["MF_Sparse10_AUC"][i]) * 100
        ax2.text(i, max(df["CKAN_Sparse10_AUC"][i], df["MF_Sparse10_AUC"][i]) + 0.03, f"+{diff:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color="#B45309")

    fig.suptitle("Hình 17. ĐÁNH GIÁ ĐA MIỀN (CROSS-DOMAIN BENCHMARK) TRÊN 3 TẬP DỮ LIỆU: MOVIE, BOOK VÀ MUSIC",
                 fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig17_path = os.path.join(FIG_DIR, "fig17_tri_dataset_benchmark.png")
    plt.savefig(fig17_path, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Đã lưu biểu đồ tổng hợp 3 dataset vào: {fig17_path}")

if __name__ == "__main__":
    run_benchmark()
