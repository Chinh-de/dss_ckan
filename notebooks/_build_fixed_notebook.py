"""Write notebooks/dss-knowledgegraph-for-rs-fixed.ipynb from the executed notebook.

Three measurement bugs are fixed and the two KG models are brought in line with the papers'
official code (hwwang55/RippleNet, weberrr/CKAN). Run from the repo root:

    backend/.venv/Scripts/python.exe notebooks/_build_fixed_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "dss-knowledgegraph-for-rs (2).ipynb"
DST = HERE / "dss-knowledgegraph-for-rs-fixed.ipynb"

nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def put(i, text):
    cells[i]["source"] = text.strip("\n").splitlines(keepends=True)


def replace(i, old, new, count=1):
    text = src(i)
    assert old in text, f"cell {i}: {old[:60]!r} not found"
    put(i, text.replace(old, new, count))


# ----------------------------------------------------------------------------- cell 0: overview
put(0, r'''
# Thực nghiệm mô hình gợi ý dựa trên Đồ thị Tri thức trên 3 tập dữ liệu (Movie, Book, Music)
So sánh 4 mô hình: MostPopular, Matrix Factorization (MF), RippleNet (CIKM 2018), CKAN (SIGIR 2020)

Mã nguồn tham chiếu (repo chính thức của hai bài báo):
- RippleNet: [hwwang55/RippleNet](https://github.com/hwwang55/RippleNet) (Hongwei Wang et al., ACM CIKM 2018)
- CKAN: [weberrr/CKAN](https://github.com/weberrr/CKAN) (Ze Wang, Guangyan Lin, Huobin Tan, Qinghong Chen, Xiyang Liu, ACM SIGIR 2020)

---

### Thay đổi so với phiên bản trước
1. **Số item lấy từ bảng tương tác**, không lấy từ cột đầu của đồ thị tri thức. Ở tập Book, bản cũ đếm 77.891 "item" trong khi chỉ có 14.967 cuốn sách, làm sai Recall@K của tập này.
2. **Thí nghiệm độ thưa dựng lại lịch sử và tập bộ ba theo từng mức dữ liệu.** Bản cũ dùng tập bộ ba dựng từ toàn bộ tập huấn luyện ở mọi mức, nên hai mô hình đồ thị thấy nhiều thông tin hơn MF.
3. **Thống kê mô tả dữ liệu chỉ tính tương tác dương** (bản cũ tính cả mẫu âm lấy mẫu 1:1).
4. **RippleNet viết lại theo repo gốc:** quan hệ là ma trận $d \times d$, cập nhật embedding item sau mỗi hop (`plus_transform`), hàm mất mát có thành phần KGE và L2, cấu hình riêng theo repo.
5. **CKAN viết lại theo repo gốc:** có lan truyền cộng tác (tập thực thể khởi đầu của item gồm các item được cùng người dùng tương tác), lan truyền tri thức và mô hình giữ nguyên cấu trúc của repo.
6. **Recall@K theo cách của repo CKAN:** ứng viên là các item xuất hiện trong train hoặc test, trừ mọi item người dùng đã gặp trong train; 100 người dùng lấy ngẫu nhiên có hạt giống cố định.

### Cấu trúc thực nghiệm
1. **Phân chia dữ liệu**: Train 60% - Validation 20% - Test 20%. Chọn trạng thái có AUC validation cao nhất rồi đánh giá trên Test.
2. **Các mô hình**: `MostPopular`, `Matrix Factorization`, `RippleNet`, `CKAN`.
3. **Thang đo**: `ROC-AUC`, `F1`, `Accuracy` (CTR); `Recall@K` với $K \in \{5, 10, 20, 50, 100\}$; ROC-AUC theo 6 tỷ lệ dữ liệu huấn luyện $10\%, 20\%, 40\%, 60\%, 80\%, 100\%$.
''')

# ----------------------------------------------------------------------------- cell 4/5: configuration
put(4, r'''
## 3. Cấu hình tham số thực nghiệm

**CKAN, MF** (theo repo weberrr/CKAN):
- `dim = 64`, `lr = 0.002`, `l2_weight = 1e-5`, `agg = 'concat'`, `item_triple_set_size = 64`.
- `user_triple_set_size`: 32 cho Movie, 16 cho Book, 8 cho Music.
- `n_layer`: 1 cho Movie, 2 cho Book và Music.

**RippleNet** (theo repo hwwang55/RippleNet; quan hệ là ma trận $d \times d$ nên số chiều nhỏ hơn):
- Movie: `dim = 16`, `n_hop = 2`, `kge_weight = 0.01`, `l2_weight = 1e-7`, `lr = 0.02`, `n_memory = 32`.
- Book: `dim = 4`, `n_hop = 2`, `kge_weight = 0.01`, `l2_weight = 1e-5`, `lr = 0.001`, `n_memory = 32`.
- Music: repo gốc không có cấu hình cho Last.FM; dùng cấu hình của Movie với `l2_weight = 1e-5`.
- `item_update_mode = 'plus_transform'`, `using_all_hops = True` (mặc định của repo).
''')

replace(5, 'active_datasets = ["movie", "book", "music"]', r'''# Cấu hình RippleNet theo repo hwwang55/RippleNet (src/main.py).
# Repo không có cấu hình cho Last.FM: tập music dùng cấu hình của movie, l2_weight như book.
RIPPLE_CONFIG = {
    "movie": {"dim": 16, "n_hop": 2, "kge_weight": 0.01, "l2_weight": 1e-7, "lr": 0.02,
              "batch_size": 1024, "n_epochs": 10, "n_memory": 32},
    "book":  {"dim": 4,  "n_hop": 2, "kge_weight": 0.01, "l2_weight": 1e-5, "lr": 0.001,
              "batch_size": 1024, "n_epochs": 10, "n_memory": 32},
    "music": {"dim": 16, "n_hop": 2, "kge_weight": 0.01, "l2_weight": 1e-5, "lr": 0.02,
              "batch_size": 1024, "n_epochs": 10, "n_memory": 32},
}
RIPPLE_ITEM_UPDATE_MODE = "plus_transform"
RIPPLE_USING_ALL_HOPS = True

active_datasets = ["movie", "book", "music"]''')

# ----------------------------------------------------------------------------- cell 10/12/13: data statistics
replace(10, "    n_i = max(int(r[:, 1].max()), int(k[:, 0].max())) + 1\n",
        "    # Item là các mã xuất hiện trong bảng tương tác; đầu của KG còn chứa thực thể không phải item.\n"
        "    n_i = int(r[:, 1].max()) + 1\n")

put(12, r'''
eda_summary = []

for ds in active_datasets:
    r_data = np.load(f"./data/{ds}/ratings_final.npy")
    k_data = np.load(f"./data/{ds}/kg_final.npy")

    # Thống kê chỉ tính tương tác dương: ratings_final còn chứa mẫu âm lấy mẫu 1:1.
    pos = r_data[r_data[:, 2] == 1]

    n_users = int(r_data[:, 0].max()) + 1
    # Item là các mã xuất hiện trong bảng tương tác (đầu của KG còn chứa thực thể không phải item).
    n_items = int(r_data[:, 1].max()) + 1
    n_ratings = len(r_data)
    n_pos = len(pos)

    sparsity = (1.0 - n_pos / (n_users * n_items)) * 100

    user_counts = list(Counter(pos[:, 0]).values())
    avg_u_inter = float(np.mean(user_counts))
    median_u_inter = float(np.median(user_counts))
    cold_start_users = sum(1 for c in user_counts if c <= 5) / n_users * 100

    item_counts = sorted(list(Counter(pos[:, 1]).values()), reverse=True)
    cum_inter = np.cumsum(item_counts) / n_pos * 100
    top20_share = cum_inter[min(int(0.2 * len(item_counts)), len(cum_inter) - 1)]

    n_entities = max(int(r_data[:, 1].max()), int(k_data[:, 0].max()), int(k_data[:, 2].max())) + 1
    n_relations = int(k_data[:, 1].max()) + 1
    n_triples = len(k_data)
    branching_factor = n_triples / n_entities
    item_degree = Counter(k_data[k_data[:, 0] < n_items][:, 0])
    median_item_triples = float(np.median([item_degree.get(i, 0) for i in range(n_items)]))

    eda_summary.append({
        "Dataset": ds.capitalize(),
        "Users": n_users,
        "Items": n_items,
        "Ratings": n_ratings,
        "Positives": n_pos,
        "Sparsity (%)": round(sparsity, 2),
        "Median_Inter": round(median_u_inter, 1),
        "Avg_Inter/User": round(avg_u_inter, 1),
        "ColdStart_Users (%)": round(cold_start_users, 1),
        "Top20%_Item_Share (%)": round(top20_share, 1),
        "KG_Entities": n_entities,
        "KG_Relations": n_relations,
        "KG_Triples": n_triples,
        "KG_Branching": round(branching_factor, 2),
        "Median_Triples/Item": median_item_triples
    })

df_eda = pd.DataFrame(eda_summary)
print(df_eda.to_string(index=False))
''')

replace(13, 'ratings = [df_eda.loc[i, "Ratings"] for i in range(len(active_datasets))]',
        'ratings = [df_eda.loc[i, "Positives"] for i in range(len(active_datasets))]')
replace(13, 'label="Ratings", color="#F59E0B")', 'label="Tương tác dương", color="#F59E0B")')
replace(13, 'user_counts_all = [list(Counter(np.load(f"./data/{ds}/ratings_final.npy")[:, 0]).values()) for ds in active_datasets]',
        'def _positive_rows(ds):\n'
        '    arr = np.load(f"./data/{ds}/ratings_final.npy")\n'
        '    return arr[arr[:, 2] == 1]\n\n'
        'user_counts_all = [list(Counter(_positive_rows(ds)[:, 0]).values()) for ds in active_datasets]')
replace(13, '    r_arr = np.load(f"./data/{ds}/ratings_final.npy")\n', '    r_arr = _positive_rows(ds)\n')

put(11, r'''
## 5. Khám phá dữ liệu (EDA)

Mọi thống kê về tương tác chỉ tính trên **tương tác dương** (`Label = 1`); số item là số mã item trong bảng tương tác.
1. **Độ thưa (Sparsity)**: tỉ lệ ô trống trong ma trận người dùng - item.
2. **Phân bố tương tác (Long-tail)**: nhóm item phổ biến chiếm bao nhiêu % lượng tương tác.
3. **User Activity**: phân bố mức độ tương tác và tỉ lệ user có $\le 5$ tương tác dương.
4. **Cấu trúc KG**: số entity, quan hệ, triple, hệ số phân nhánh và số triple trung vị của mỗi item.
''')

# ----------------------------------------------------------------------------- cell 16/17/18: RippleNet
put(16, r'''
## PHẦN B: MÔ HÌNH RIPPLENET (CIKM 2018)

Bài báo: **RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems**
Tác giả: *Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo*
Mã nguồn tham chiếu: [hwwang55/RippleNet](https://github.com/hwwang55/RippleNet). Phần cài đặt dưới đây chuyển mã TensorFlow của repo sang PyTorch và giữ nguyên các phép tính.

---

### 1. Ripple set
Tập hạt giống là các item người dùng đã tương tác trong tập huấn luyện: $\mathcal{E}_u^0 = \mathcal{V}_u$. Với $k = 1, \dots, H$:
$$\mathcal{S}_u^k = \{(h, r, t) \mid (h, r, t) \in \mathcal{G},\ h \in \mathcal{E}_u^{k-1}\}, \qquad \mathcal{E}_u^k = \{t \mid (h, r, t) \in \mathcal{S}_u^k\}$$
Mỗi hop lấy mẫu cố định `n_memory` bộ ba (`get_ripple_set` trong `data_loader.py`).

### 2. Lan truyền sở thích (`_key_addressing`)
Quan hệ là ma trận $\mathbf{R}_i \in \mathbb{R}^{d \times d}$. Với embedding item $\mathbf{v}$:
$$p_i = \text{softmax}\left(\mathbf{v}^\top \mathbf{R}_i \mathbf{h}_i\right), \qquad \mathbf{o}_u^k = \sum_i p_i\, \mathbf{t}_i$$
Sau mỗi hop, embedding item được cập nhật theo `item_update_mode = plus_transform`:
$$\mathbf{v} \leftarrow (\mathbf{v} + \mathbf{o}_u^k)\, \mathbf{M}$$

### 3. Dự đoán (`predict`, `using_all_hops = True`)
$$\mathbf{u} = \sum_{k=1}^{H} \mathbf{o}_u^k, \qquad \hat{y}_{uv} = \sigma\left(\mathbf{v}^\top \mathbf{u}\right)$$

### 4. Hàm mất mát (`_build_loss`)
$$\mathcal{L} = \text{BCE}(y, \hat{y}) \;-\; \lambda_{kge} \sum_{k} \text{mean}\left(\sigma(\mathbf{h}^\top \mathbf{R}\, \mathbf{t})\right) \;+\; \lambda_{l2} \sum_{k} \left(\lVert\mathbf{h}\rVert^2 + \lVert\mathbf{t}\rVert^2 + \lVert\mathbf{R}\rVert^2\right)$$
''')

put(17, r'''
# ============================================================
# LAN TRUYỀN CỘNG TÁC VÀ LAN TRUYỀN TRI THỨC
# Theo data_loader.py của weberrr/CKAN; get_ripple_set của hwwang55/RippleNet
# dùng cùng một thuật toán cho phía người dùng.
# ============================================================
class KnowledgeGraph:
    # Đồ thị tri thức ở dạng CSR: các bộ ba của thực thể e nằm ở vị trí indptr[e] .. indptr[e+1]-1,
    # theo đúng thứ tự xuất hiện trong kg_final (như kg[head].append((tail, relation)) của repo).
    def __init__(self, kg_np, n_entity):
        order = np.argsort(kg_np[:, 0], kind="stable")
        heads = kg_np[order, 0].astype(np.int64)
        self.relations = kg_np[order, 1].astype(np.int64)
        self.tails = kg_np[order, 2].astype(np.int64)
        self.indptr = np.zeros(n_entity + 1, dtype=np.int64)
        np.cumsum(np.bincount(heads, minlength=n_entity), out=self.indptr[1:])


def construct_kg(kg_np, n_entity=None):
    if n_entity is None:
        n_entity = int(max(kg_np[:, 0].max(), kg_np[:, 2].max())) + 1
    return KnowledgeGraph(kg_np, n_entity)


def collaboration_propagation(train_data, n_item):
    # Tập thực thể khởi đầu (CKAN, mục 3.1.1).
    # - Người dùng: các item đã tương tác dương trong train.
    # - Item: các item được cùng người dùng tương tác; item chưa có ai tương tác thì lấy chính nó.
    user_history_item_dict = defaultdict(list)
    item_history_user_dict = defaultdict(list)
    for user, item, rating in train_data:
        if rating == 1:
            user_history_item_dict[int(user)].append(int(item))
            item_history_user_dict[int(item)].append(int(user))

    item_neighbor_item_dict = dict()
    for item, users in item_history_user_dict.items():
        neighbors = set()
        for user in users:
            neighbors.update(user_history_item_dict[user])
        item_neighbor_item_dict[item] = list(neighbors)

    for item in range(n_item):
        if item not in item_neighbor_item_dict:
            item_neighbor_item_dict[item] = [item]
    return dict(user_history_item_dict), item_neighbor_item_dict


def kg_propagation(kg, init_entity_set, set_size, n_layer):
    # triple_sets[obj][layer] = (heads, relations, tails), mỗi danh sách dài set_size.
    # Tầng 0 xuất phát từ tập khởi đầu, tầng sau xuất phát từ các đuôi của tầng trước; mỗi tầng lấy
    # mẫu đều set_size bộ ba trong số mọi bộ ba có đầu thuộc tập thực thể (lặp lại nếu không đủ).
    # Kết quả giống kg_propagation của repo; khác ở chỗ không liệt kê hết các bộ ba mà lấy mẫu
    # thẳng theo chỉ số, nhờ đó nhanh hơn nhiều khi tập khởi đầu của item có hàng nghìn phần tử.
    # Tầng rỗng thì lặp lại tầng trước (như repo). Repo không xử lý trường hợp tầng 0 rỗng;
    # ở đây tầng đó được đệm bằng một vòng tự nối của thực thể hạt giống đầu tiên.
    triple_sets = defaultdict(list)
    indptr = kg.indptr
    for obj, seeds in init_entity_set.items():
        for l in range(n_layer):
            entities = np.asarray(seeds if l == 0 else triple_sets[obj][-1][2], dtype=np.int64)
            degree = indptr[entities + 1] - indptr[entities]
            total = int(degree.sum())

            if total == 0:
                if triple_sets[obj]:
                    triple_sets[obj].append(triple_sets[obj][-1])
                else:
                    seed = int(entities[0])
                    triple_sets[obj].append(([seed] * set_size, [0] * set_size, [seed] * set_size))
            else:
                indices = np.random.choice(total, size=set_size, replace=(total < set_size))
                ends = np.cumsum(degree)
                which = np.searchsorted(ends, indices, side="right")
                positions = indptr[entities[which]] + (indices - (ends[which] - degree[which]))
                triple_sets[obj].append((
                    entities[which].tolist(),
                    kg.relations[positions].tolist(),
                    kg.tails[positions].tolist(),
                ))
    return triple_sets
''')

put(18, r'''
# ============================================================
# Mô hình RippleNet (chuyển từ TensorFlow của hwwang55/RippleNet sang PyTorch)
# ============================================================
class RippleNet(nn.Module):
    def __init__(self, n_entity, n_relation, dim=16, n_hop=2, kge_weight=0.01, l2_weight=1e-7,
                 item_update_mode="plus_transform", using_all_hops=True):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_hop = n_hop
        self.kge_weight = kge_weight
        self.l2_weight = l2_weight
        self.item_update_mode = item_update_mode
        self.using_all_hops = using_all_hops

        self.entity_emb = nn.Embedding(n_entity, dim)
        # Mỗi quan hệ là một ma trận dim x dim (lưu phẳng)
        self.relation_emb = nn.Embedding(n_relation, dim * dim)
        # Ma trận biến đổi dùng khi cập nhật embedding item cuối mỗi hop
        self.transform_matrix = nn.Parameter(torch.empty(dim, dim))

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight.view(n_relation, dim, dim))
        nn.init.xavier_uniform_(self.transform_matrix)

    def _lookup(self, memories_h, memories_r, memories_t):
        h_emb_list, r_emb_list, t_emb_list = [], [], []
        for hop in range(self.n_hop):
            # [batch, n_memory, dim]
            h_emb_list.append(self.entity_emb(memories_h[hop]))
            # [batch, n_memory, dim, dim]
            r_emb_list.append(self.relation_emb(memories_r[hop]).view(-1, memories_r[hop].shape[1], self.dim, self.dim))
            # [batch, n_memory, dim]
            t_emb_list.append(self.entity_emb(memories_t[hop]))
        return h_emb_list, r_emb_list, t_emb_list

    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            return o
        if self.item_update_mode == "plus":
            return item_embeddings + o
        if self.item_update_mode == "replace_transform":
            return torch.matmul(o, self.transform_matrix)
        if self.item_update_mode == "plus_transform":
            return torch.matmul(item_embeddings + o, self.transform_matrix)
        raise Exception("Unknown item updating mode: " + self.item_update_mode)

    def _key_addressing(self, item_embeddings, h_emb_list, r_emb_list, t_emb_list):
        o_list = []
        for hop in range(self.n_hop):
            # [batch, n_memory, dim]
            Rh = torch.matmul(r_emb_list[hop], h_emb_list[hop].unsqueeze(3)).squeeze(3)
            # [batch, n_memory]
            probs = torch.matmul(Rh, item_embeddings.unsqueeze(2)).squeeze(2)
            probs_normalized = F.softmax(probs, dim=1)
            # [batch, dim]
            o = (t_emb_list[hop] * probs_normalized.unsqueeze(2)).sum(dim=1)
            item_embeddings = self._update_item_embedding(item_embeddings, o)
            o_list.append(o)
        return o_list, item_embeddings

    def forward(self, items, memories_h, memories_r, memories_t):
        """Trả về logit (chưa qua sigmoid) và các embedding dùng cho hàm mất mát."""
        item_embeddings = self.entity_emb(items)
        h_emb_list, r_emb_list, t_emb_list = self._lookup(memories_h, memories_r, memories_t)
        o_list, item_embeddings = self._key_addressing(item_embeddings, h_emb_list, r_emb_list, t_emb_list)

        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y = y + o_list[i]
        scores = (item_embeddings * y).sum(dim=1)
        return scores, (h_emb_list, r_emb_list, t_emb_list)

    def compute_loss(self, scores, labels, emb_lists):
        h_emb_list, r_emb_list, t_emb_list = emb_lists
        base_loss = F.binary_cross_entropy_with_logits(scores, labels)

        kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = h_emb_list[hop].unsqueeze(2)   # [batch, n_memory, 1, dim]
            t_expanded = t_emb_list[hop].unsqueeze(3)   # [batch, n_memory, dim, 1]
            hRt = torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded).squeeze()
            kge_loss = kge_loss + torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss = l2_loss + (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss = l2_loss + (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss = l2_loss + (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        return base_loss + kge_loss + l2_loss


class RippleNetTrainer:
    """Vòng huấn luyện và đánh giá của RippleNet (train.py của repo), tối ưu bằng Adam."""
    def __init__(self, model, lr=0.02, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_memories(self, users, ripple_set):
        memories_h, memories_r, memories_t = [], [], []
        for hop in range(self.model.n_hop):
            memories_h.append(torch.LongTensor([ripple_set[int(u)][hop][0] for u in users]).to(self.device))
            memories_r.append(torch.LongTensor([ripple_set[int(u)][hop][1] for u in users]).to(self.device))
            memories_t.append(torch.LongTensor([ripple_set[int(u)][hop][2] for u in users]).to(self.device))
        return memories_h, memories_r, memories_t

    def train_epoch(self, train_data, ripple_set, batch_size=1024):
        self.model.train()
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)
        total_loss, n_batch = 0.0, 0
        for s in range(0, len(train_data), batch_size):
            b = train_data[indices[s:s + batch_size]]
            items = torch.LongTensor(b[:, 1]).to(self.device)
            labels = torch.FloatTensor(b[:, 2]).to(self.device)
            scores, emb_lists = self.model(items, *self._get_memories(b[:, 0], ripple_set))
            loss = self.model.compute_loss(scores, labels, emb_lists)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        return total_loss / max(n_batch, 1)

    def predict(self, data, ripple_set, batch_size=1024):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for s in range(0, len(data), batch_size):
                b = data[s:s + batch_size]
                items = torch.LongTensor(b[:, 1]).to(self.device)
                logits, _ = self.model(items, *self._get_memories(b[:, 0], ripple_set))
                scores.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(scores)

    def evaluate(self, data, ripple_set, batch_size=1024):
        scores = self.predict(data, ripple_set, batch_size)
        labels = data[:, 2]
        auc = roc_auc_score(labels, scores)
        pred_bin = (scores >= 0.5).astype(int)
        f1 = f1_score(labels, pred_bin, zero_division=0)
        acc = accuracy_score(labels, pred_bin)
        return auc, f1, acc, scores

    def score_all_items(self, user, ripple_set, n_item, batch_size=2048):
        """Điểm của mọi item cho một người dùng (biểu diễn người dùng phụ thuộc item nên phải chạy mô hình)."""
        pairs = np.stack([np.full(n_item, user), np.arange(n_item), np.zeros(n_item, dtype=np.int64)], axis=1)
        return self.predict(pairs, ripple_set, batch_size)
''')

# ----------------------------------------------------------------------------- cell 19..22: CKAN
put(19, r'''
## PHẦN C: MÔ HÌNH CKAN (SIGIR 2020)

Bài báo: **CKAN: Collaborative Knowledge-aware Attentive Network for Recommender Systems**
Tác giả: *Ze Wang, Guangyan Lin, Huobin Tan, Qinghong Chen, Xiyang Liu*
Mã nguồn tham chiếu: [weberrr/CKAN](https://github.com/weberrr/CKAN). Lớp mô hình dưới đây giữ nguyên `model.py` của repo; phần dựng tập bộ ba ở trên theo `data_loader.py`.

---

### 1. Lan truyền dị thể
1. **Lan truyền cộng tác** (`collaboration_propagation`): tập thực thể khởi đầu của người dùng là các item họ đã tương tác; của item là các item được cùng người dùng tương tác.
$$\mathcal{E}_u^0 = \{e \mid (v, e) \in \mathcal{A},\ y_{uv} = 1\}, \qquad \mathcal{E}_v^0 = \{e \mid (v_u, e) \in \mathcal{A},\ v_u \in \{v_u \mid \exists u:\ y_{uv} = 1 \wedge y_{uv_u} = 1\}\}$$
2. **Lan truyền tri thức** (`kg_propagation`): $\mathcal{S}_o^l = \{(h, r, t) \mid (h, r, t) \in \mathcal{G},\ h \in \mathcal{E}_o^{l-1}\}$, lấy mẫu cố định số bộ ba mỗi tầng.

### 2. Embedding có chú ý nhận biết tri thức (`_knowledge_attention`)
$$\pi(\mathbf{e}_h, \mathbf{r}) = \sigma\left(\mathbf{W}_2\, \text{ReLU}\left(\mathbf{W}_1\, \text{ReLU}\left(\mathbf{W}_0 [\mathbf{e}_h \Vert \mathbf{r}]\right)\right)\right), \qquad \mathbf{e}_o^{(l)} = \sum_i \text{softmax}(\pi_i)\, \mathbf{e}_{t_i}$$

### 3. Tổng hợp và dự đoán (`predict`)
- Người dùng: trung bình embedding các thực thể đầu của tầng 1 (tức tập khởi đầu) và các biểu diễn tầng.
- Item: embedding gốc của item và các biểu diễn tầng (với `sum`, `pool` có thêm trung bình tập khởi đầu).
$$\hat{y}_{uv} = \sigma\left(\mathbf{e}_u^\top \mathbf{e}_v\right)$$
''')

put(20, r'''
# ============================================================
# Mô hình CKAN (model.py của weberrr/CKAN)
# ============================================================
class CKAN(nn.Module):
    def __init__(self, n_entity, n_relation, dim=64, n_layer=1, agg="concat"):
        super(CKAN, self).__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_layer = n_layer
        self.agg = agg

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.attention = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(),
            nn.Linear(self.dim, 1, bias=False),
            nn.Sigmoid(),
        )
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def _user_embeddings(self, user_triple_set):
        # user_triple_set = [h, r, t], mỗi phần là danh sách theo tầng các tensor [batch, triple_set_size]
        user_embeddings = []
        # [batch, triple_set_size, dim] -> [batch, dim]
        user_emb_0 = self.entity_emb(user_triple_set[0][0])
        user_embeddings.append(user_emb_0.mean(dim=1))
        for i in range(self.n_layer):
            h_emb = self.entity_emb(user_triple_set[0][i])
            r_emb = self.relation_emb(user_triple_set[1][i])
            t_emb = self.entity_emb(user_triple_set[2][i])
            user_embeddings.append(self._knowledge_attention(h_emb, r_emb, t_emb))
        return user_embeddings

    def _item_embeddings(self, items, item_triple_set):
        item_embeddings = []
        # [batch, dim]
        item_emb_origin = self.entity_emb(items)
        item_embeddings.append(item_emb_origin)
        for i in range(self.n_layer):
            h_emb = self.entity_emb(item_triple_set[0][i])
            r_emb = self.relation_emb(item_triple_set[1][i])
            t_emb = self.entity_emb(item_triple_set[2][i])
            item_embeddings.append(self._knowledge_attention(h_emb, r_emb, t_emb))
        if self.n_layer > 0 and (self.agg == "sum" or self.agg == "pool"):
            # [batch, triple_set_size, dim] -> [batch, dim]
            item_emb_0 = self.entity_emb(item_triple_set[0][0])
            item_embeddings.append(item_emb_0.mean(dim=1))
        return item_embeddings

    def forward(self, items, user_triple_set, item_triple_set):
        user_embeddings = self._user_embeddings(user_triple_set)
        item_embeddings = self._item_embeddings(items, item_triple_set)
        return self.predict(user_embeddings, item_embeddings)

    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]

        if self.agg == "concat":
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u), dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v), dim=-1)
        elif self.agg == "sum":
            for i in range(1, len(user_embeddings)):
                e_u = e_u + user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v = e_v + item_embeddings[i]
        elif self.agg == "pool":
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)

        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores

    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        att_weights = self.attention(torch.cat((h_emb, r_emb), dim=-1)).squeeze(-1)
        # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights, dim=-1)
        # [batch_size, triple_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        # [batch_size, dim]
        emb_i = emb_i.sum(dim=1)
        return emb_i

    # --------------------------------------------------------
    # Hai hàm dưới không có trong repo. Chúng tách biểu diễn người dùng và item để xếp hạng
    # nhanh trên toàn danh mục và để xuất embedding item. Thứ tự ghép là [tầng 0, tầng 1, ...]
    # cho cả hai phía, nên sigmoid(e_u . e_v) trùng với kết quả của forward().
    # --------------------------------------------------------
    def _aggregate(self, embeddings):
        if self.agg == "concat":
            return torch.cat(embeddings, dim=-1)
        if self.agg == "sum":
            return torch.stack(embeddings, dim=0).sum(dim=0)
        return torch.stack(embeddings, dim=0).max(dim=0).values

    def get_user_embeddings(self, user_triple_set):
        with torch.no_grad():
            return self._aggregate(self._user_embeddings(user_triple_set))

    def get_item_embeddings(self, items, item_triple_set):
        with torch.no_grad():
            return self._aggregate(self._item_embeddings(items, item_triple_set))
''')

put(21, r'''
# ============================================================
# Huấn luyện và đánh giá CKAN (train.py của weberrr/CKAN)
# ============================================================
class CKANTrainer:
    def __init__(self, model, lr=0.002, weight_decay=1e-5, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.criterion = nn.BCELoss()

    def get_triple_tensor(self, objs, triple_set):
        # [h, r, t], mỗi phần: danh sách theo tầng các tensor [batch_size, triple_set_size]
        h, r, t = [], [], []
        for i in range(self.model.n_layer):
            h.append(torch.LongTensor([triple_set[int(obj)][i][0] for obj in objs]).to(self.device))
            r.append(torch.LongTensor([triple_set[int(obj)][i][1] for obj in objs]).to(self.device))
            t.append(torch.LongTensor([triple_set[int(obj)][i][2] for obj in objs]).to(self.device))
        return [h, r, t]

    def _feed(self, data, user_triple_set, item_triple_set):
        items = torch.LongTensor(data[:, 1]).to(self.device)
        users_triple = self.get_triple_tensor(data[:, 0], user_triple_set)
        items_triple = self.get_triple_tensor(data[:, 1], item_triple_set)
        return items, users_triple, items_triple

    def train_epoch(self, train_data, user_triple_set, item_triple_set, batch_size=2048):
        self.model.train()
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)
        total_loss, n_batch = 0.0, 0
        for s in range(0, len(train_data), batch_size):
            b = train_data[indices[s:s + batch_size]]
            labels = torch.FloatTensor(b[:, 2]).to(self.device)
            scores = self.model(*self._feed(b, user_triple_set, item_triple_set))
            loss = self.criterion(scores, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        return total_loss / max(n_batch, 1)

    def evaluate(self, data, user_triple_set, item_triple_set, batch_size=4096):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for s in range(0, len(data), batch_size):
                b = data[s:s + batch_size]
                scores.append(self.model(*self._feed(b, user_triple_set, item_triple_set)).cpu().numpy())
        scores = np.concatenate(scores)
        labels = data[:, 2]
        # AUC, F1 tính trên toàn bộ tập (repo lấy trung bình theo từng batch)
        auc = roc_auc_score(labels, scores)
        pred_bin = (scores >= 0.5).astype(int)
        f1 = f1_score(labels, pred_bin, zero_division=0)
        acc = accuracy_score(labels, pred_bin)
        return auc, f1, acc, scores
''')

put(22, r'''
# ============================================================
# Embedding item của CKAN cho toàn bộ danh mục (không phụ thuộc người dùng)
# ============================================================
def ckan_item_matrix(trainer, item_triple_set, n_item, chunk_size=2048):
    """Ma trận [n_item, D]; hàng i là biểu diễn của item i."""
    trainer.model.eval()
    chunks = []
    for start in range(0, n_item, chunk_size):
        ids = list(range(start, min(start + chunk_size, n_item)))
        items = torch.LongTensor(ids).to(trainer.device)
        triples = trainer.get_triple_tensor(ids, item_triple_set)
        chunks.append(trainer.model.get_item_embeddings(items, triples))
    return torch.cat(chunks, dim=0)
''')

put(23, r'''
# ============================================================
# Đánh giá xếp hạng Top-K (topk_eval trong train.py của weberrr/CKAN)
# ============================================================
class TopKRecommenderEvaluator:
    """Recall@K theo cách của repo CKAN:
    - Ứng viên: các item xuất hiện trong train hoặc test, trừ mọi item người dùng đã gặp trong train
      (cả dương lẫn âm).
    - Đáp án: các item dương của người dùng trong test.
    - 100 người dùng lấy ngẫu nhiên trong số người có mặt ở cả train và test. Repo không cố định
      hạt giống ở bước này; ở đây cố định để cả bốn mô hình được đo trên cùng một nhóm người dùng.
    """
    def __init__(self, train_data, test_data, n_item, k_list=(5, 10, 20, 50, 100), user_num=100, seed=42):
        self.k_list = list(k_list)
        self.n_item = n_item

        self.item_mask = np.zeros(n_item, dtype=bool)
        self.item_mask[np.unique(np.concatenate([train_data[:, 1], test_data[:, 1]]))] = True

        self.train_record = defaultdict(set)
        for user, item, _ in train_data:
            self.train_record[int(user)].add(int(item))
        self.test_record = defaultdict(set)
        for user, item, label in test_data:
            if label == 1:
                self.test_record[int(user)].add(int(item))

        user_list = sorted(set(self.train_record.keys()) & set(self.test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.RandomState(seed).choice(user_list, size=user_num, replace=False).tolist()
        self.user_list = [int(u) for u in user_list]

    def evaluate_model(self, score_func):
        """score_func(user) -> mảng điểm dài n_item."""
        recall_list = {k: [] for k in self.k_list}
        max_k = max(self.k_list)
        for user in self.user_list:
            scores = np.asarray(score_func(user), dtype=np.float64).copy()
            candidates = self.item_mask.copy()
            candidates[list(self.train_record[user])] = False
            scores[~candidates] = -np.inf

            top = np.argpartition(-scores, min(max_k, len(scores) - 1))[:max_k]
            top = top[np.argsort(-scores[top])]
            positives = self.test_record[user]
            for k in self.k_list:
                recall_list[k].append(len(set(top[:k].tolist()) & positives) / len(positives))
        return {f"Recall@{k}": float(np.mean(recall_list[k])) if recall_list[k] else 0.0 for k in self.k_list}
''')

put(24, r'''
## PHẦN D: THỰC NGHIỆM ĐÁNH GIÁ (MOVIE, BOOK, MUSIC)

1. **Chia dữ liệu 6:2:2**, giữ lại các người dùng có ít nhất một tương tác dương trong train (như hai repo).
2. **Huấn luyện 4 mô hình**; MF, RippleNet, CKAN chọn trạng thái có `val_auc` cao nhất.
   - CKAN: tập khởi đầu của người dùng và item dựng bằng lan truyền cộng tác trên train.
   - RippleNet: ripple set dựng từ lịch sử dương trong train, cấu hình riêng theo repo.
3. **Đo lường**
   - CTR: `ROC-AUC`, `F1`, `Accuracy` trên test.
   - Top-K: `Recall@5, 10, 20, 50, 100` trên 100 người dùng, theo cách của repo CKAN.
   - Độ thưa: giữ lại $10\%$ đến $100\%$ tập train. Ở mỗi mức, **lịch sử người dùng và các tập bộ ba được dựng lại chỉ từ phần dữ liệu được giữ**, nên cả ba mô hình thấy cùng một lượng thông tin. Để so sánh được giữa các mức, tập test của thí nghiệm này chỉ gồm những người dùng đã có tương tác dương ngay ở mức $10\%$.
''')

# ----------------------------------------------------------------------------- cell 25: benchmark
put(25, r'''
# ============================================================
# THỰC NGHIỆM BENCHMARK (chia 6:2:2, chọn mô hình theo validation)
# ============================================================
import copy

def evaluate_predictions(labels, scores):
    auc = roc_auc_score(labels, scores)
    pred_bin = (scores >= 0.5).astype(int)
    f1 = f1_score(labels, pred_bin, zero_division=0)
    acc = accuracy_score(labels, pred_bin)
    return auc, f1, acc


def mf_predict(model, data, batch_size):
    model.eval()
    with torch.no_grad():
        out = [model(torch.LongTensor(b[:, 0]).to(device), torch.LongTensor(b[:, 1]).to(device)).cpu().numpy()
               for b in (data[s:s + batch_size] for s in range(0, len(data), batch_size))]
    return np.concatenate(out)


def train_mf(train_rows, eval_rows, n_user, n_item, cfg, batch_size):
    """Huấn luyện MF, trả về mô hình ở trạng thái có AUC validation cao nhất."""
    model = MatrixFactorizationBaseline(n_user, n_item, dim=cfg["dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    criterion = nn.BCELoss()
    best_auc, best_state = -1.0, None
    for _ in range(cfg["n_epochs"]):
        model.train()
        order = np.random.permutation(len(train_rows))
        for s in range(0, len(train_rows), batch_size):
            b = train_rows[order[s:s + batch_size]]
            optimizer.zero_grad()
            pred = model(torch.LongTensor(b[:, 0]).to(device), torch.LongTensor(b[:, 1]).to(device))
            criterion(pred, torch.FloatTensor(b[:, 2]).to(device)).backward()
            optimizer.step()
        v_auc, _, _ = evaluate_predictions(eval_rows[:, 2], mf_predict(model, eval_rows, batch_size))
        if v_auc > best_auc:
            best_auc, best_state = v_auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_auc


def train_ripplenet(train_rows, eval_rows, ripple_set, n_entity, n_relation, rcfg, batch_size):
    model = RippleNet(n_entity, n_relation, dim=rcfg["dim"], n_hop=rcfg["n_hop"],
                      kge_weight=rcfg["kge_weight"], l2_weight=rcfg["l2_weight"],
                      item_update_mode=RIPPLE_ITEM_UPDATE_MODE, using_all_hops=RIPPLE_USING_ALL_HOPS)
    trainer = RippleNetTrainer(model, lr=rcfg["lr"], device=device)
    best_auc, best_state = -1.0, None
    for _ in range(rcfg["n_epochs"]):
        trainer.train_epoch(train_rows, ripple_set, batch_size=batch_size)
        v_auc, _, _, _ = trainer.evaluate(eval_rows, ripple_set, batch_size=rcfg["batch_size"])
        if v_auc > best_auc:
            best_auc, best_state = v_auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return trainer, best_auc


def train_ckan(train_rows, eval_rows, user_triple_set, item_triple_set, n_entity, n_relation, cfg, batch_size):
    model = CKAN(n_entity, n_relation, dim=cfg["dim"], n_layer=cfg["n_layer"], agg=cfg["agg"])
    trainer = CKANTrainer(model, lr=cfg["lr"], weight_decay=cfg["l2_weight"], device=device)
    best_auc, best_state = -1.0, None
    for _ in range(cfg["n_epochs"]):
        trainer.train_epoch(train_rows, user_triple_set, item_triple_set, batch_size=batch_size)
        v_auc, _, _, _ = trainer.evaluate(eval_rows, user_triple_set, item_triple_set)
        if v_auc > best_auc:
            best_auc, best_state = v_auc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return trainer, best_auc


def run_comprehensive_benchmark(ds_name):
    print(f"\n--- DATASET: {ds_name.upper()} ---")

    cfg = CONFIG[ds_name]
    rcfg = RIPPLE_CONFIG[ds_name]
    rating_np = np.load(f"./data/{ds_name}/ratings_final.npy")
    kg_np = np.load(f"./data/{ds_name}/kg_final.npy")

    n_user = int(rating_np[:, 0].max()) + 1
    # Item là các mã trong bảng tương tác. Đầu của KG còn chứa thực thể không phải item
    # (tập book: 14.967 item nhưng mã đầu lớn nhất của KG là 77.890).
    n_item = int(rating_np[:, 1].max()) + 1
    n_entity = max(int(rating_np[:, 1].max()), int(kg_np[:, 0].max()), int(kg_np[:, 2].max())) + 1
    n_relation = int(kg_np[:, 1].max()) + 1
    kg = construct_kg(kg_np, n_entity)

    print(f"Users: {n_user:,} | Items: {n_item:,} | Ratings: {len(rating_np):,} | Triples: {len(kg_np):,}")

    # ------------------------------------------------------------
    # CHIA DỮ LIỆU 6:2:2
    # ------------------------------------------------------------
    np.random.seed(42)
    shuffled_ratings = rating_np.copy()
    np.random.shuffle(shuffled_ratings)
    n_samples = len(shuffled_ratings)
    train_end = int(n_samples * 0.6)
    eval_end = int(n_samples * 0.8)

    train_data = shuffled_ratings[:train_end]
    eval_data = shuffled_ratings[train_end:eval_end]
    test_data = shuffled_ratings[eval_end:]

    # Như hai repo: chỉ giữ người dùng có ít nhất một tương tác dương trong train
    valid_train_users = np.unique(train_data[train_data[:, 2] == 1][:, 0])
    train_data = train_data[np.isin(train_data[:, 0], valid_train_users)]
    eval_data = eval_data[np.isin(eval_data[:, 0], valid_train_users)]
    test_data = test_data[np.isin(test_data[:, 0], valid_train_users)]

    print(f"Phân chia: Train={len(train_data):,} | Eval={len(eval_data):,} | Test={len(test_data):,}")

    topk_evaluator = TopKRecommenderEvaluator(train_data, test_data, n_item, k_list=[5, 10, 20, 50, 100])
    bs = cfg["batch_size"]

    # ------------------------------------------------------------
    # 1. MOST POPULAR
    # ------------------------------------------------------------
    pop = MostPopularBaseline()
    pop.fit(train_data, n_item)
    pop_scores = pop.predict(test_data[:, 0], test_data[:, 1])
    pop_auc, pop_f1, pop_acc = evaluate_predictions(test_data[:, 2], pop_scores)
    pop_topk = topk_evaluator.evaluate_model(lambda u: pop.score_all_items(u, n_item))
    print(f"  [1/4] MostPopular       : Test_AUC={pop_auc:.4f} | F1={pop_f1:.4f} | ACC={pop_acc:.4f} | Rec@10={pop_topk['Recall@10']:.4f} | Rec@50={pop_topk['Recall@50']:.4f}")

    # ------------------------------------------------------------
    # 2. MATRIX FACTORIZATION
    # ------------------------------------------------------------
    mf, best_mf_val_auc = train_mf(train_data, eval_data, n_user, n_item, cfg, bs)
    mf_auc, mf_f1, mf_acc = evaluate_predictions(test_data[:, 2], mf_predict(mf, test_data, bs))
    mf_topk = topk_evaluator.evaluate_model(lambda u: mf.score_all_items(u, device))
    print(f"  [2/4] MF                : Val_AUC={best_mf_val_auc:.4f} | Test_AUC={mf_auc:.4f} | F1={mf_f1:.4f} | ACC={mf_acc:.4f} | Rec@10={mf_topk['Recall@10']:.4f} | Rec@50={mf_topk['Recall@50']:.4f}")

    # ------------------------------------------------------------
    # LAN TRUYỀN CỘNG TÁC VÀ LAN TRUYỀN TRI THỨC (chỉ dùng train)
    # ------------------------------------------------------------
    user_init_entity_set, item_init_entity_set = collaboration_propagation(train_data, n_item)
    user_triple_set = kg_propagation(kg, user_init_entity_set, cfg["utss"], cfg["n_layer"])
    item_triple_set = kg_propagation(kg, item_init_entity_set, cfg["itss"], cfg["n_layer"])
    ripple_set = kg_propagation(kg, user_init_entity_set, rcfg["n_memory"], rcfg["n_hop"])

    # ------------------------------------------------------------
    # 3. RIPPLENET
    # ------------------------------------------------------------
    ripple_trainer, best_rip_val_auc = train_ripplenet(train_data, eval_data, ripple_set, n_entity, n_relation, rcfg, rcfg["batch_size"])
    ripple_auc, ripple_f1, ripple_acc, _ = ripple_trainer.evaluate(test_data, ripple_set, batch_size=rcfg["batch_size"])
    ripple_topk = topk_evaluator.evaluate_model(lambda u: ripple_trainer.score_all_items(u, ripple_set, n_item))
    print(f"  [3/4] RippleNet         : Val_AUC={best_rip_val_auc:.4f} | Test_AUC={ripple_auc:.4f} | F1={ripple_f1:.4f} | ACC={ripple_acc:.4f} | Rec@10={ripple_topk['Recall@10']:.4f} | Rec@50={ripple_topk['Recall@50']:.4f}")

    # ------------------------------------------------------------
    # 4. CKAN
    # ------------------------------------------------------------
    ckan_trainer, best_ckan_val_auc = train_ckan(train_data, eval_data, user_triple_set, item_triple_set, n_entity, n_relation, cfg, bs)
    ckan = ckan_trainer.model
    ckan_auc, ckan_f1, ckan_acc, _ = ckan_trainer.evaluate(test_data, user_triple_set, item_triple_set)

    # Embedding item không phụ thuộc người dùng: tính một lần cho toàn bộ danh mục
    all_item_matrix = ckan_item_matrix(ckan_trainer, item_triple_set, n_item)

    def ckan_score_fn(u):
        u_emb = ckan.get_user_embeddings(ckan_trainer.get_triple_tensor([u], user_triple_set))
        return torch.sigmoid(torch.matmul(u_emb, all_item_matrix.T)).squeeze(0).cpu().numpy()

    ckan_topk = topk_evaluator.evaluate_model(ckan_score_fn)
    print(f"  [4/4] CKAN              : Val_AUC={best_ckan_val_auc:.4f} | Test_AUC={ckan_auc:.4f} | F1={ckan_f1:.4f} | ACC={ckan_acc:.4f} | Rec@10={ckan_topk['Recall@10']:.4f} | Rec@50={ckan_topk['Recall@50']:.4f}")

    # ------------------------------------------------------------
    # LƯU MÔ HÌNH CKAN VÀ CÁC TỆP ĐI KÈM (dùng cho hệ thống minh hoạ)
    # ------------------------------------------------------------
    save_dir = f"./saved_models/{ds_name}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(ckan.state_dict(), os.path.join(save_dir, "ckan_model.pt"))
    np.save(os.path.join(save_dir, "item_embeddings.npy"), all_item_matrix.cpu().numpy())
    model_config = {
        "dataset": ds_name,
        "n_user": int(n_user), "n_item": int(n_item),
        "n_entity": int(n_entity), "n_relation": int(n_relation),
        "dim": int(cfg["dim"]), "n_layer": int(cfg["n_layer"]),
        "itss": int(cfg["itss"]), "utss": int(cfg["utss"]),
        "agg": str(cfg["agg"])
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    with open(os.path.join(save_dir, "item_triple_set.pkl"), "wb") as f:
        pickle.dump(dict(item_triple_set), f)
    with open(os.path.join(save_dir, "user_history.pkl"), "wb") as f:
        pickle.dump({u: set(items) for u, items in user_init_entity_set.items()}, f)
    map_file = f"./data/{ds_name}/item_index2entity_id.txt"
    if os.path.exists(map_file):
        shutil.copy(map_file, os.path.join(save_dir, "item_index2entity_id.txt"))

    # ------------------------------------------------------------
    # THỰC NGHIỆM ĐỘ THƯA: 6 mức dữ liệu huấn luyện, đo ROC-AUC
    # Ở mỗi mức, lịch sử người dùng và các tập bộ ba được dựng lại chỉ từ phần train được giữ.
    # Tập eval/test của thí nghiệm gồm những người dùng đã có tương tác dương ở mức nhỏ nhất,
    # nên cả 6 mức và cả 3 mô hình được đo trên cùng một tập.
    # ------------------------------------------------------------
    sparsity_ratios = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    mf_sparsity_aucs, ripple_sparsity_aucs, ckan_sparsity_aucs = [], [], []

    smallest = train_data[:max(int(len(train_data) * sparsity_ratios[0]), 128)]
    sparse_users = np.unique(smallest[smallest[:, 2] == 1][:, 0])
    eval_sp = eval_data[np.isin(eval_data[:, 0], sparse_users)]
    test_sp = test_data[np.isin(test_data[:, 0], sparse_users)]
    print(f"  Sparsity ({len(sparse_users):,} người dùng có lịch sử ở mức 10%; test dùng {len(test_sp):,} mẫu):")

    for r in sparsity_ratios:
        sub_tr = train_data[:max(int(len(train_data) * r), 128)]
        # Như hai repo: bỏ các dòng của người dùng chưa có tương tác dương trong phần được giữ
        sub_users = np.unique(sub_tr[sub_tr[:, 2] == 1][:, 0])
        sub_tr = sub_tr[np.isin(sub_tr[:, 0], sub_users)]
        # Co batch trên tập con nhỏ để vẫn đủ số bước cập nhật
        sub_bs = min(bs, max(128, len(sub_tr) // 8))
        sub_rbs = min(rcfg["batch_size"], max(128, len(sub_tr) // 8))

        # 1. MF
        m_sp, _ = train_mf(sub_tr, eval_sp, n_user, n_item, cfg, sub_bs)
        mf_auc_r, _, _ = evaluate_predictions(test_sp[:, 2], mf_predict(m_sp, test_sp, bs))
        mf_sparsity_aucs.append(mf_auc_r)

        # Tập khởi đầu và tập bộ ba chỉ dựng từ phần train được giữ
        sub_user_init, sub_item_init = collaboration_propagation(sub_tr, n_item)
        sub_user_triple = kg_propagation(kg, sub_user_init, cfg["utss"], cfg["n_layer"])
        sub_item_triple = kg_propagation(kg, sub_item_init, cfg["itss"], cfg["n_layer"])
        sub_ripple_set = kg_propagation(kg, sub_user_init, rcfg["n_memory"], rcfg["n_hop"])

        # 2. RippleNet
        rip_tr_sp, _ = train_ripplenet(sub_tr, eval_sp, sub_ripple_set, n_entity, n_relation, rcfg, sub_rbs)
        rip_auc_r, _, _, _ = rip_tr_sp.evaluate(test_sp, sub_ripple_set, batch_size=rcfg["batch_size"])
        ripple_sparsity_aucs.append(rip_auc_r)

        # 3. CKAN
        ck_tr_sp, _ = train_ckan(sub_tr, eval_sp, sub_user_triple, sub_item_triple, n_entity, n_relation, cfg, sub_bs)
        ck_auc_r, _, _, _ = ck_tr_sp.evaluate(test_sp, sub_user_triple, sub_item_triple)
        ckan_sparsity_aucs.append(ck_auc_r)

        print(f"    - Mốc {int(r*100):3d}% Train: MF={mf_auc_r:.4f} | RippleNet={rip_auc_r:.4f} | CKAN={ck_auc_r:.4f} ")

    return {
        "summary": {
            "Dataset": ds_name.capitalize(),
            "Users": n_user, "Items": n_item, "Ratings": len(rating_np), "KG_Triples": len(kg_np),
            "MostPop_AUC": pop_auc, "MF_AUC": mf_auc, "Ripple_AUC": ripple_auc, "CKAN_AUC": ckan_auc,
            "MostPop_F1": pop_f1, "MF_F1": mf_f1, "Ripple_F1": ripple_f1, "CKAN_F1": ckan_f1,
            "MostPop_ACC": pop_acc, "MF_ACC": mf_acc, "Ripple_ACC": ripple_acc, "CKAN_ACC": ckan_acc,
            "MF_Rec10": mf_topk['Recall@10'], "Ripple_Rec10": ripple_topk['Recall@10'], "CKAN_Rec10": ckan_topk['Recall@10'],
            "MF_Rec50": mf_topk['Recall@50'], "Ripple_Rec50": ripple_topk['Recall@50'], "CKAN_Rec50": ckan_topk['Recall@50'],
            "MF_Sparse10_AUC": mf_sparsity_aucs[0], "Ripple_Sparse10_AUC": ripple_sparsity_aucs[0], "CKAN_Sparse10_AUC": ckan_sparsity_aucs[0]
        },
        "topk": {
            "MostPop": pop_topk,
            "MF": mf_topk,
            "RippleNet": ripple_topk,
            "CKAN": ckan_topk
        },
        "sparsity": {
            "ratios": sparsity_ratios,
            "MF": mf_sparsity_aucs,
            "RippleNet": ripple_sparsity_aucs,
            "CKAN": ckan_sparsity_aucs,
            "n_users": int(len(sparse_users)),
            "n_test": int(len(test_sp))
        },
        "split": {"ratings": int(len(rating_np)), "train": int(len(train_data)), "eval": int(len(eval_data)), "test": int(len(test_data))}
    }

# Chạy thực nghiệm trên cả 3 tập dữ liệu
benchmark_results = {}
for ds in active_datasets:
    benchmark_results[ds] = run_comprehensive_benchmark(ds)

# Lưu toàn bộ kết quả ra tệp để dùng lại (báo cáo, hệ thống minh hoạ)
with open("./saved_models/benchmark_results_raw.json", "w", encoding="utf-8") as f:
    json.dump(benchmark_results, f, indent=2, default=float)

# Tổng hợp bảng so sánh kết quả
df_results = pd.DataFrame([benchmark_results[ds]["summary"] for ds in active_datasets])
print("\n" + "="*80)
print("TỔNG HỢP HIỆU SUẤT CTR VÀ ĐỘ THỬA (3 DATASETS)")
print("="*80)
display_cols = ["Dataset", "Ratings", "MF_AUC", "Ripple_AUC", "CKAN_AUC", "MF_F1", "Ripple_F1", "CKAN_F1", "MF_Sparse10_AUC", "CKAN_Sparse10_AUC"]
print(df_results[display_cols].to_string(index=False))
''')

# Cell 3's diagram describes the item branch as starting from the item alone; say what the code does now.
replace(3, "- **Item Branch**: Lan truyền ngữ nghĩa từ item ứng viên sang các láng giềng trên KG.",
        "- **Item Branch**: Lan truyền từ tập khởi đầu của item (các item được cùng người dùng tương tác) sang các láng giềng trên KG.")

# ----------------------------------------------------------------------------- clear outputs and write
for cell in cells:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", DST.name, f"({len(cells)} cells)")
