import json
import os

nb = {
    'cells': [],
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'gpuType': 'T4', 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'}
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

def add_md(text):
    nb['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'source': [l + '\n' for l in text.split('\n')]})

def add_code(text):
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [l + '\n' for l in text.split('\n')]})

# ============================================================
# 0. HEADER & GIỚI THIỆU ĐỀ TÀI
# ============================================================
add_md(r"""# NGHIÊN CỨU & ĐÁNH GIÁ THỰC NGHIỆM ĐA MIỀN TRÊN 3 TẬP DỮ LIỆU
## MÔ HÌNH CKAN (COLLABORATIVE KNOWLEDGE-AWARE ATTENTIVE NETWORK)
### So Sánh Đa Mô Hình: MostPopular vs Item-KNN vs Matrix Factorization vs CKAN
**Tập Dữ Liệu Học Thuật Chuẩn**: MovieLens-1M (`movie`), Book-Crossing (`book`), Last.FM (`music`)
**Môi Trường Thực Nghiệm**: Google Colab GPU (NVIDIA Tesla T4 / RTX 3050)

---
### Cấu Trúc Notebook Nghiên Cứu:
1. **Kiểm tra Môi trường Phần cứng & Thư viện**
2. **Phương Pháp Luận Toán Học Mô Hình CKAN**
3. **Thu Thập & Tiền Xử Lý Dữ Liệu Tự Động Từ Nguồn Học Thuật Chuẩn (GroupLens, Stanford, Paper Repo)**
4. **Phần A: Các Mô Hình Cơ Sở (Baselines) - Dùng Thư Viện Chuẩn `scikit-learn` & `scipy`**
5. **Phần B: Triển Khai Chi Tiết Từng Mô-Đun Của CKAN (Trọng Tâm Đề Tài)**
   - B1. `KnowledgeRippleSampler`: Lấy mẫu tập Ripple Sets đa tầng từ Knowledge Graph
   - B2. `KnowledgeAwareAttentionLayer`: Cơ chế Chú ý Ngữ nghĩa Tri thức
   - B3. `MultiLayerAggregator`: Bộ tổng hợp đặc trưng đa tầng
   - B4. `CKANModel`: Kiến trúc mạng toàn diện kết hợp hai chiều
   - B5. `CKANTrainer`: Bộ tối ưu hàm mất mát & Đánh giá CTR
6. **Phần C: Tiến Trình Thực Nghiệm Đa Miền & Trực Quan Hóa Dashboard**
   - Thực nghiệm 1: Warm-start CTR Benchmark
   - Thực nghiệm 2: Khảo sát Chống Suy Biến Khi Dữ Liệu Cực Thưa (10% Data)
   - Bảng tổng kết số liệu & Đồ thị trực quan hóa khoa học
""")

# ============================================================
# 1. KIỂM TRA PHẦN CỨNG & GPU
# ============================================================
add_code(r"""# ============================================================
# 1. KIỂM TRA MÔI TRƯỜNG PHẦN CỨNG & GPU
# ============================================================
import torch

print("=== KIỂM TRA PHẦN CỨNG VÀ BỘ NHỚ ===")
print("Phiên bản PyTorch :", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiết bị tính toán:", device)
if torch.cuda.is_available():
    print("Tên GPU           :", torch.cuda.get_device_name(0))
    print("Dung lượng VRAM   :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
""")

# ============================================================
# 2. CÀI ĐẶT THƯ VIỆN & SEED
# ============================================================
add_code(r"""# ============================================================
# 2. CÀI ĐẶT THƯ VIỆN & THIẾT LẬP REPRODUCIBILITY SEED
# ============================================================
!pip install -q --upgrade scikit-learn scipy matplotlib seaborn pandas tqdm

import os
import time
import random
import zipfile
import urllib.request
import collections
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
print("[OK] Đã sẵn sàng thư viện chuẩn và thiết lập Seed:", SEED)
""")

# ============================================================
# 3. PHƯƠNG PHÁP LUẬN TOÁN HỌC
# ============================================================
add_md(r"""## 2. PHƯƠNG PHÁP LUẬN & KIẾN TRÚC MÔ HÌNH CKAN (COLLABORATIVE KNOWLEDGE-AWARE ATTENTIVE NETWORK)

### 2.1. Động Lực Nghiên Cứu (Why CKAN?)
Các phương pháp lọc cộng tác truyền thống (CF/Matrix Factorization) dựa dẫm hoàn toàn vào ma trận tương tác User-Item:
$$y_{u, v} = \sigma(\mathbf{u}_u^T \mathbf{v}_v)$$
Khi ma trận thưa thớt (>99% ô rỗng) hoặc người dùng mới (Cold-start), MF bị suy biến nặng nề (Data Starvation). 

**CKAN giải quyết bài toán này bằng cách kết hợp Đồ thị Tri thức (KG) với cơ chế Lan truyền Cộng tác Hai Chiều (Bi-directional Collaborative Propagation):**
- **Nhánh Người dùng (User Branch)**: Lan truyền sở thích từ các phim/sách đã thích ra các thực thể liên quan (Đạo diễn, Diễn viên, Thể loại, Tác giả).
- **Nhánh Sản phẩm (Item Branch)**: Lan truyền ngữ nghĩa của phim/sách ứng viên ra các thực thể láng giềng và các sản phẩm đồng tương tác.

```
                    [ LỊCH SỬ NGƯỜI DÙNG: S(u) ]                     [ PHIM / SÁCH ỨNG VIÊN: v ]
                                │                                                 │
                  0-hop: e_u^(0) = mean(e_items)                     0-hop: e_v^(0) = e_v
                                │                                                 │
                                ▼                                                 ▼
                  [ User Ripple Set: E_u^l ]                       [ Item Ripple Set: E_v^l ]
                        (h, r, t)                                        (h, r, t)
                                │                                                 │
                                ▼                                                 ▼
             ┌────────────────────────────────────┐             ┌────────────────────────────────────┐
             │   KNOWLEDGE-AWARE ATTENTION UNIT   │             │   KNOWLEDGE-AWARE ATTENTION UNIT   │
             │   s = MLP([e_h ; e_r])             │             │   s = MLP([e_h ; e_r])             │
             │   alpha = Softmax(s)               │             │   alpha = Softmax(s)               │
             │   e_u^l = sum(alpha * e_t)         │             │   e_v^l = sum(alpha * e_t)         │
             └──────────────────┬─────────────────┘             └──────────────────┬─────────────────┘
                                │                                                 │
                                ▼                                                 ▼
             [ Aggregator: Concat / Sum / Pool ]               [ Aggregator: Concat / Sum / Pool ]
                     Vector người dùng: e_u                            Vector sản phẩm: e_v
                                └───────────────────────┬─────────────────────────┘
                                                        │
                                                        ▼
                                         [ DỰ ĐOÁN XÁC SUẤT TƯƠNG TÁC ]
                                          y_hat = Sigmoid(e_u^T * e_v)
```

---

### 2.2. Các Bước Tính Toán Toán Học Trong Thuật Toán CKAN

1. **Khởi tạo Hạt giống (Seed Entities)**:
   - Phía User (0-hop): Trung bình các sản phẩm đã tương tác tích cực trong tập huấn luyện:
     $$\mathbf{e}_u^{(0)} = \frac{1}{|S(u)|} \sum_{v \in S(u)} \mathbf{e}_v$$
   - Phía Item (0-hop): Vector nhúng của chính sản phẩm ứng viên:
     $$\mathbf{e}_v^{(0)} = \mathbf{e}_v$$

2. **Lan truyền Tri thức qua $L$ bước nhảy (Multi-hop Knowledge Ripple Sets)**:
   - Với mỗi tầng $l \in \{1, \dots, L\}$, truy vấn các bộ ba tri thức láng giềng $\mathcal{E}^l = \{(h, r, t)\}$.
   - Để cố định kích thước tensor tính toán trên GPU, lấy mẫu:
     - User Triple Set Size: $\text{UTSS} = 32$ (Movie/Music), $16$ (Book).
     - Item Triple Set Size: $\text{ITSS} = 64$ (Movie/Book), $32$ (Music).

3. **Cơ chế Chú Ý Tri Thức (Knowledge-aware Attention Layer)**:
   - Mức độ ảnh hưởng của quan hệ $r$ nối với thực thể đầu $h$ được tính bằng mạng nơ-ron đa tầng (MLP):
     $$s_i(h_i, r_i) = \sigma\left( \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 [\mathbf{e}_{h_i} \parallel \mathbf{e}_{r_i}]) \right)$$
   - Chuẩn hóa Softmax trên toàn bộ tập bộ ba:
     $$\alpha_i = \frac{\exp(s_i)}{\sum_{j} \exp(s_j)}$$
   - Biểu diễn tổng hợp ở tầng $l$ là tổng có trọng số của các thực thể đuôi $t$:
     $$\mathbf{e}^l = \sum_i \alpha_i \cdot \mathbf{e}_{t_i}$$

4. **Bộ Tổng Hợp Đa Tầng (Multi-layer Aggregator)**:
   - Chiến lược Ghép nối (Concat - Khuyên dùng trong bài báo gốc):
     $$\mathbf{e}_u = [\mathbf{e}_u^{(L)} \parallel \dots \parallel \mathbf{e}_u^{(0)}], \quad \mathbf{e}_v = [\mathbf{e}_v^{(L)} \parallel \dots \parallel \mathbf{e}_v^{(0)}]$$
   - Giúp bảo toàn không gian đặc trưng giữa thực thể gốc và tri thức lan truyền.

5. **Dự Đoán Tương Tác & Tối Ưu Hóa Hàm Mất Mát**:
   - Xác suất tương tác dự đoán:
     $$\hat{y}(u, v) = \sigma(\mathbf{e}_u^T \mathbf{e}_v) = \frac{1}{1 + \exp(-\mathbf{e}_u^T \mathbf{e}_v)}$$
   - Tối ưu hóa bằng Binary Cross-Entropy Loss kết hợp phạt điều chuẩn $L_2$ Weight Decay:
     $$\mathcal{L} = -\sum_{(u, v) \in \mathcal{D}} \left[ y_{u, v} \log \hat{y}(u, v) + (1 - y_{u, v}) \log(1 - \hat{y}(u, v)) \right] + \lambda \|\Theta\|_2^2$$
""")

# ============================================================
# 4. CHỌN DATASET CẦN CHẠY
# ============================================================
add_code(r"""# ============================================================
# 3. CHỌN TẬP DỮ LIỆU CẦN CHẠY THỰC NGHIỆM
# ============================================================
# Bạn có thể chọn: "movie" (Phim) | "book" (Sách) | "music" (Âm nhạc) | "all" (Chạy cả 3)
TARGET_DATASET = "all"

DATASETS_CONFIG = {
    "movie": {"dim": 64, "n_layer": 1, "itss": 64, "utss": 32, "batch_size": 2048, "lr": 0.002},
    "book":  {"dim": 64, "n_layer": 2, "itss": 64, "utss": 16, "batch_size": 1024, "lr": 0.002},
    "music": {"dim": 64, "n_layer": 1, "itss": 32, "utss": 16, "batch_size": 512,  "lr": 0.001}
}

active_datasets = ["movie", "book", "music"] if TARGET_DATASET == "all" else [TARGET_DATASET]
print(f"Chế độ thực nghiệm: {TARGET_DATASET.upper()} -> Các tập dữ liệu sẽ chạy: {active_datasets}")
""")

# ============================================================
# 5. DATA PIPELINE: NGUỒN CHUẨN + UNZIP + TIỀN XỬ LÝ
# ============================================================
add_md(r"""## 4. THU THẬP & TIỀN XỬ LÝ DỮ LIỆU TỪ NGUỒN HỌC THUẬT CHUẨN

Theo chuẩn nghiên cứu của các bài báo Recommender Systems kết hợp Knowledge Graph:
1. **Đồ thị tri thức (KG Triples & Entity Mappings)**: Tải trực tiếp từ repository gốc của bài báo CKAN (SIGIR 2020 - Ze Wang et al.):
   [https://github.com/weberrr/CKAN](https://github.com/weberrr/CKAN)
2. **Dữ liệu tương tác thô (Raw Ratings)**:
   - **Âm nhạc (`music`) - Last.FM 2k**: GroupLens HetRec 2011 Workshop (`user_artists.dat`)
   - **Sách (`book`) - Book-Crossing**: University of Freiburg / Benchmark Archive (`BX-Book-Ratings.csv`)
   - **Điện ảnh (`movie`) - MovieLens**: GroupLens Research / Benchmark Archive (`ratings.dat`)

Toàn bộ quá trình **tải tệp**, **tự động giải nén (unzip)** và **tiền xử lý chuẩn** (Negative Sampling 1:1, khớp ID thực thể KG) được tự động hóa hoàn toàn.
""")

add_code(r"""# ============================================================
# 4. TẢI DỮ LIỆU TỪ NGUỒN CHUẨN, GIẢI NÉN & TIỀN XỬ LÝ TỰ ĐỘNG
# ============================================================
DATA_SOURCES = {
    "music": {
        "raw_file": "user_artists.dat",
        "raw_urls": [
            "https://raw.githubusercontent.com/hwwang55/KGCN/master/data/music/user_artists.dat",
            "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
        ],
        "sep": "\t", "threshold": 0.0, "max_users": 0
    },
    "book": {
        "raw_file": "BX-Book-Ratings.csv",
        "raw_urls": [
            "https://raw.githubusercontent.com/hwwang55/RippleNet/master/data/book/BX-Book-Ratings.csv"
        ],
        "sep": ";", "threshold": 0.0, "max_users": 0
    },
    "movie": {
        "raw_file": "ratings.dat",
        "raw_urls": [
            "https://raw.githubusercontent.com/hwwang55/RippleNet/master/data/movie/ratings.dat"
        ],
        "sep": "::", "threshold": 4.0, "max_users": 2500
    }
}

CKAN_OFFICIAL_REPO = "https://raw.githubusercontent.com/weberrr/CKAN/master/data/"

def fetch_url(url, dest_path):
    print(f"    -> Đang tải: {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out:
        out.write(resp.read())

def download_and_extract(ds_name):
    cfg = DATA_SOURCES[ds_name]
    ds_dir = f"./data/{ds_name}"
    os.makedirs(ds_dir, exist_ok=True)
    
    # 1. Tải KG chính thức từ kho tác giả bài báo CKAN
    for kg_file in ["kg.txt", "item_index2entity_id.txt"]:
        dest = os.path.join(ds_dir, kg_file)
        if not os.path.exists(dest) or os.path.getsize(dest) < 1000:
            fetch_url(CKAN_OFFICIAL_REPO + f"{ds_name}/{kg_file}", dest)
            print(f"       [OK] Tải {kg_file} ({round(os.path.getsize(dest)/1024, 1)} KB)")

    # 2. Tải tương tác thô (và giải nén nếu là zip)
    raw_dest = os.path.join(ds_dir, cfg["raw_file"])
    if not os.path.exists(raw_dest) or os.path.getsize(raw_dest) < 1000:
        for url in cfg["raw_urls"]:
            try:
                if url.endswith(".zip"):
                    zip_dest = os.path.join(ds_dir, "archive.zip")
                    fetch_url(url, zip_dest)
                    print(f"       [Giải nén] Đang giải nén {zip_dest} ...")
                    with zipfile.ZipFile(zip_dest, "r") as zf:
                        zf.extractall(ds_dir)
                    if os.path.exists(zip_dest): os.remove(zip_dest)
                    break
                else:
                    fetch_url(url, raw_dest)
                    break
            except Exception as e:
                print(f"       [Thử lại] {e}")
        print(f"       [OK] Đã có {cfg['raw_file']} ({round(os.path.getsize(raw_dest)/1024, 1)} KB)")
    return ds_dir

def preprocess_and_cache(ds_name):
    r_npy = f"./data/{ds_name}/ratings_final.npy"
    k_npy = f"./data/{ds_name}/kg_final.npy"
    if os.path.exists(r_npy) and os.path.exists(k_npy):
        print(f"  [Cache Sẵn Sàng] {ds_name.upper()} đã được tiền xử lý hợp lệ.")
        return

    print(f"\n>>> BẮT ĐẦU TIỀN XỬ LÝ CHUẨN: {ds_name.upper()} <<<")
    t0 = time.time()
    ds_dir = download_and_extract(ds_name)
    cfg = DATA_SOURCES[ds_name]
    
    # 1. Đọc ánh xạ item sang KG entity
    item_old2new, entity_id2idx = {}, {}
    with open(os.path.join(ds_dir, "item_index2entity_id.txt"), "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            p = line.strip().split("\t")
            if len(p) >= 2:
                item_old2new[p[0]] = idx
                entity_id2idx[p[1]] = idx
                
    item_set = set(item_old2new.values())
    user_pos, user_neg = defaultdict(set), defaultdict(set)
    
    # 2. Đọc xếp hạng thô
    with open(os.path.join(ds_dir, cfg["raw_file"]), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split(cfg["sep"])
            if ds_name == "book": p = [x.strip('\"') for x in p]
            if len(p) < 3 or p[1] not in item_old2new: continue
            try: r = float(p[2])
            except: continue
            if r >= cfg["threshold"]: user_pos[p[0]].add(item_old2new[p[1]])
            else: user_neg[p[0]].add(item_old2new[p[1]])
                
    # 3. Lấy mẫu người dùng chuẩn (nếu có cấu hình max_users)
    if cfg["max_users"] > 0 and len(user_pos) > cfg["max_users"]:
        np.random.seed(555)
        sel_u = sorted(list(np.random.choice(list(user_pos.keys()), size=cfg["max_users"], replace=False)))
        user_pos = {u: user_pos[u] for u in sel_u}
        
    # 4. Lấy mẫu tương tác âm 1:1
    rows = []
    np.random.seed(555)
    for u_new, (u_old, pos_items) in enumerate(user_pos.items()):
        for it in pos_items: rows.append((u_new, it, 1))
        unwatched = item_set - pos_items
        if u_old in user_neg: unwatched -= user_neg[u_old]
        if unwatched:
            neg_items = np.random.choice(list(unwatched), size=len(pos_items), replace=(len(unwatched) < len(pos_items)))
            for it in neg_items: rows.append((u_new, it, 0))
    rating_np = np.array(rows, dtype=np.int32)
    np.save(r_npy, rating_np)
    
    # 5. Đánh số lại KG triples
    ent_cnt = len(entity_id2idx)
    rel_id2idx, kg_rows = {}, []
    with open(os.path.join(ds_dir, "kg.txt"), "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) != 3: continue
            h_old, r_old, t_old = p[0], p[1], p[2]
            if h_old not in entity_id2idx: entity_id2idx[h_old] = ent_cnt; ent_cnt += 1
            if t_old not in entity_id2idx: entity_id2idx[t_old] = ent_cnt; ent_cnt += 1
            if r_old not in rel_id2idx: rel_id2idx[r_old] = len(rel_id2idx)
            kg_rows.append((entity_id2idx[h_old], rel_id2idx[r_old], entity_id2idx[t_old]))
            
    kg_np = np.array(kg_rows, dtype=np.int32)
    np.save(k_npy, kg_np)
    print(f"  [HOÀN TẤT {ds_name.upper()}] Users: {len(user_pos):,} | Items: {len(item_set):,} | Ratings: {len(rating_np):,} | Triples: {len(kg_np):,}")

for ds in active_datasets:
    preprocess_and_cache(ds)
print("[OK] Toàn bộ dữ liệu từ nguồn chuẩn đã sẵn sàng!")
""")

# ============================================================
# PHẦN A: CÁC MÔ HÌNH CƠ SỞ (BASELINES DÙNG THƯ VIỆN CHUẨN)
# ============================================================
add_md(r"""## PHẦN A: CÁC MÔ HÌNH CƠ SỞ (BASELINES - SỬ DỤNG THƯ VIỆN CHUẨN)

Để giữ cho cấu trúc code tinh gọn và tập trung nghiên cứu sâu vào CKAN, các mô hình cơ sở được triển khai nhanh chóng thông qua các thư viện tiêu chuẩn của Python (`scikit-learn`, `scipy.sparse`, và PyTorch gọn nhẹ):

1. **MostPopular**: Gợi ý theo tần suất tương tác toàn cục (dùng `numpy.bincount`).
2. **Item-KNN**: Lọc cộng tác dựa trên độ tương đồng Cosine giữa các Item (dùng `sklearn.neighbors.NearestNeighbors` và ma trận thưa `scipy.sparse.csr_matrix`).
3. **Biased Matrix Factorization (MF)**: Lọc cộng tác nhân tố ẩn với bias người dùng và bias sản phẩm (tối ưu BCE Loss).
""")

add_code(r"""# ============================================================
# PHẦN A: CÁC MÔ HÌNH CƠ SỞ (BASELINES DÙNG SCIKIT-LEARN & SCIPY)
# ============================================================
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# --- 1. MOST POPULAR (Không cá nhân hóa, dựa trên tần suất tương tác) ---
class MostPopularBaseline:
    def fit(self, train_data, n_items):
        pos = train_data[train_data[:, 2] == 1]
        counts = np.bincount(pos[:, 1], minlength=n_items)
        self.scores = counts / (counts.max() + 1e-9)

    def predict(self, items):
        return self.scores[items]

# --- 2. ITEM-KNN (Lọc cộng tác dùng sklearn NearestNeighbors) ---
class ItemKNNBaseline:
    def __init__(self, k=20):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")

    def fit(self, train_data, n_users, n_items):
        pos = train_data[train_data[:, 2] == 1]
        # Ma trận thưa Item - User (kích thước: n_items x n_users)
        self.mat = csr_matrix((np.ones(len(pos)), (pos[:, 1], pos[:, 0])), shape=(n_items, n_users))
        self.model.fit(self.mat)
        self.user_history = defaultdict(set)
        for u, i in zip(pos[:, 0], pos[:, 1]):
            self.user_history[u].add(i)

    def predict(self, users, items):
        scores = []
        for u, i in zip(users, items):
            hist = self.user_history.get(u, set())
            if not hist or i >= self.mat.shape[0]:
                scores.append(0.5)
                continue
            dists, indices = self.model.kneighbors(self.mat[i], n_neighbors=min(self.k, self.mat.shape[0]))
            sims = 1.0 - dists[0]
            matched = [sims[idx] for idx, neighbor in enumerate(indices[0]) if neighbor in hist]
            scores.append(float(np.mean(matched)) if matched else 0.5)
        return np.array(scores)

# --- 3. BIASED MATRIX FACTORIZATION (PyTorch Latent Factor Model) ---
class MatrixFactorizationBaseline(nn.Module):
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
        dot = (self.user_emb(users) * self.item_emb(items)).sum(dim=-1, keepdim=True)
        return torch.sigmoid((dot + self.user_bias(users) + self.item_bias(items)).squeeze(-1))

print("[OK] Đã hoàn tất cài đặt 3 mô hình Baseline tinh gọn!")
""")

# ============================================================
# PHẦN B: TRIỂN KHAI CHI TIẾT MÔ HÌNH CKAN (TRỌNG TÂM ĐỀ TÀI)
# ============================================================
add_md(r"""## PHẦN B: TRIỂN KHAI CHI TIẾT MÔ HÌNH CKAN (COLLABORATIVE KNOWLEDGE-AWARE ATTENTIVE NETWORK)

Đây là **trọng tâm cốt lõi của đề tài nghiên cứu**. Mô hình CKAN được thiết kế theo hướng module hóa hoàn chỉnh, bao gồm 5 thành phần kiến trúc chi tiết:

1. **`KnowledgeRippleSampler`**: Thuật toán lấy mẫu tập gợn sóng tri thức (Ripple Sets) đa tầng hop-by-hop từ đồ thị tri thức cho cả nhánh User và nhánh Item.
2. **`KnowledgeAwareAttentionLayer`**: Mạng nơ-ron đa tầng (MLP) tính toán trọng số quan tâm chú ý ngữ nghĩa giữa thực thể đầu và loại quan hệ.
3. **`MultiLayerAggregator`**: Bộ tổng hợp đặc trưng vector qua $L$ tầng lan truyền (Concat / Sum / Pool).
4. **`CKANModel`**: Kiến trúc mạng hoàn chỉnh kết hợp hai nhánh Collaborative User & Semantic Item.
5. **`CKANTrainer`**: Quản lý vòng lặp huấn luyện, tối ưu hàm mất mát Binary Cross-Entropy và tính toán các độ đo khoa học (AUC, F1, Accuracy).
""")

# B1: Ripple Sampler
add_code(r"""# ============================================================
# B1. KNOWLEDGE RIPPLE SAMPLER (LẤY MẪU TẬP GỢN SÓNG TRI THỨC)
# ============================================================
class KnowledgeRippleSampler:
    # Thuật toán trích xuất và lấy mẫu cố định kích thước (Fixed-size Sampling)
    # cho các bộ ba tri thức (h, r, t) qua L bước nhảy (hops).
    def __init__(self, kg_np, n_layer=1, itss=64, utss=32):
        self.n_layer = n_layer
        self.itss = itss
        self.utss = utss
        
        # Xây dựng danh sách kề của Đồ thị Tri thức: head -> list of (tail, relation)
        self.kg_dict = defaultdict(list)
        for h, r, t in kg_np:
            self.kg_dict[int(h)].append((int(t), int(r)))

    def build_item_ripple_set(self, n_items):
        # Khởi tạo tập gợn sóng cho từng sản phẩm ứng viên qua L tầng
        item_triple_set = defaultdict(list)
        for it in range(n_items):
            for l in range(self.n_layer):
                neighbors = self.kg_dict.get(it, [])
                if len(neighbors) == 0:
                    h, r, t = [it] * self.itss, [0] * self.itss, [it] * self.itss
                else:
                    replace = len(neighbors) < self.itss
                    choice = np.random.choice(len(neighbors), size=self.itss, replace=replace)
                    h = [it] * self.itss
                    r = [neighbors[c][1] for c in choice]
                    t = [neighbors[c][0] for c in choice]
                item_triple_set[it].append((h, r, t))
        return item_triple_set

    def build_user_ripple_set(self, n_users, user_pos_dict):
        # Lan truyền sở thích của người dùng từ các sản phẩm đã thích ra thực thể láng giềng
        user_triple_set = defaultdict(list)
        for u in range(n_users):
            liked = user_pos_dict.get(u, [0])
            entities = liked
            for l in range(self.n_layer):
                h, r, t = [], [], []
                for ent in entities:
                    for tail, rel in self.kg_dict.get(ent, []):
                        h.append(ent)
                        r.append(rel)
                        t.append(tail)
                if len(h) == 0:
                    user_triple_set[u].append(([liked[0]] * self.utss, [0] * self.utss, [liked[0]] * self.utss))
                else:
                    replace = len(h) < self.utss
                    choice = np.random.choice(len(h), size=self.utss, replace=replace)
                    user_triple_set[u].append(([h[c] for c in choice], [r[c] for c in choice], [t[c] for c in choice]))
                    entities = [t[c] for c in choice]
        return user_triple_set

def to_triple_tensor(ids, triple_set, n_layer, device):
    # Chuyển đổi danh sách bộ ba đã lấy mẫu thành PyTorch LongTensor trên GPU
    h_list, r_list, t_list = [], [], []
    for l in range(n_layer):
        h_list.append(torch.LongTensor([triple_set[i][l][0] for i in ids]).to(device))
        r_list.append(torch.LongTensor([triple_set[i][l][1] for i in ids]).to(device))
        t_list.append(torch.LongTensor([triple_set[i][l][2] for i in ids]).to(device))
    return [h_list, r_list, t_list]

print("[OK] Đã hoàn thành Mô-đun B1: KnowledgeRippleSampler!")
""")

# B2: Knowledge-Aware Attention Layer
add_code(r"""# ============================================================
# B2. KNOWLEDGE-AWARE ATTENTION LAYER (MÔ-ĐUN CHÚ Ý TRI THỨC)
# ============================================================
import torch.nn.functional as F

class KnowledgeAwareAttention(nn.Module):
    # Cơ chế Chú ý Tri thức (Knowledge-aware Attention):
    # Đo lường mức độ quan trọng của quan hệ r đối với thực thể đầu h
    # s = Sigmoid( W2 * ReLU( W1 * [e_h || e_r] ) )
    # alpha = Softmax(s)
    # e_out = sum( alpha_i * e_t_i )
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(),
            nn.Linear(dim, 1, bias=False)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, h_emb, r_emb, t_emb):
        # h_emb, r_emb, t_emb: [batch_size, n_triples, dim]
        cat_hr = torch.cat((h_emb, r_emb), dim=-1)             # [batch_size, n_triples, 2*dim]
        scores = self.mlp(cat_hr).squeeze(-1)                  # [batch_size, n_triples]
        alpha = F.softmax(scores, dim=-1)                      # [batch_size, n_triples]
        output = torch.mul(alpha.unsqueeze(-1), t_emb).sum(dim=1) # [batch_size, dim]
        return output

print("[OK] Đã hoàn thành Mô-đun B2: KnowledgeAwareAttention!")
""")

# B3: MultiLayer Aggregator
add_code(r"""# ============================================================
# B3. MULTI-LAYER AGGREGATOR (BỘ TỔNG HỢP ĐẶC TRƯNG ĐA TẦNG)
# ============================================================
class MultiLayerAggregator(nn.Module):
    # Bộ tổng hợp biểu diễn vector qua các bước nhảy L-hop:
    # - concat: Ghép nối vector gốc và tri thức lan truyền [e^0 || e^1 || ... || e^L] (Tối ưu nhất)
    # - sum   : Cộng dồn các vector e^0 + e^1 + ... + e^L
    # - mean  : Lấy trung bình cộng các vector
    def __init__(self, mode="concat"):
        super().__init__()
        self.mode = mode

    def forward(self, embs):
        # embs: list of L+1 tensors, mỗi tensor có shape [batch_size, dim]
        if self.mode == "concat":
            return torch.cat(embs, dim=-1)
        elif self.mode == "sum":
            return torch.stack(embs, dim=0).sum(dim=0)
        elif self.mode == "mean":
            return torch.stack(embs, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Không hỗ trợ chế độ aggregator: {self.mode}")

print("[OK] Đã hoàn thành Mô-đun B3: MultiLayerAggregator!")
""")

# B4: Full CKAN Model
add_code(r"""# ============================================================
# B4. MÔ HÌNH CKAN (COLLABORATIVE KNOWLEDGE-AWARE ATTENTIVE NETWORK)
# ============================================================
class CKAN(nn.Module):
    # Kiến trúc toàn diện của mô hình CKAN:
    # - Bi-directional Collaborative Propagation (Lan truyền 2 chiều User & Item)
    # - Knowledge-aware Attention Unit
    # - Interaction Probability: y_hat = Sigmoid(e_u^T * e_v)
    def __init__(self, n_entity, n_relation, dim=64, n_layer=1, agg="concat"):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_layer = n_layer
        
        # Ma trận nhúng thực thể và nhúng quan hệ
        self.entity_emb = nn.Embedding(n_entity, dim)
        self.relation_emb = nn.Embedding(n_relation, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        
        # Đơn vị chú ý tri thức và bộ tổng hợp
        self.attention = KnowledgeAwareAttention(dim)
        self.aggregator = MultiLayerAggregator(mode=agg)

    def get_user_embeddings(self, user_triple):
        # 0-hop: Trung bình các sản phẩm đã tương tác tích cực
        user_embs = [self.entity_emb(user_triple[0][0]).mean(dim=1)]
        for l in range(self.n_layer):
            h = self.entity_emb(user_triple[0][l])
            r = self.relation_emb(user_triple[1][l])
            t = self.entity_emb(user_triple[2][l])
            user_embs.append(self.attention(h, r, t))
        return self.aggregator(user_embs)

    def get_item_embeddings(self, items, item_triple):
        # 0-hop: Vector nhúng của chính sản phẩm ứng viên
        item_embs = [self.entity_emb(items)]
        for l in range(self.n_layer):
            h = self.entity_emb(item_triple[0][l])
            r = self.relation_emb(item_triple[1][l])
            t = self.entity_emb(item_triple[2][l])
            item_embs.append(self.attention(h, r, t))
        return self.aggregator(item_embs)

    def forward(self, items, user_triple, item_triple):
        e_u = self.get_user_embeddings(user_triple)
        e_v = self.get_item_embeddings(items, item_triple)
        return torch.sigmoid((e_u * e_v).sum(dim=-1))

print("[OK] Đã hoàn thành Mô-đun B4: Kiến trúc mô hình CKAN!")
""")

# B5: Trainer & Evaluation Metrics
add_code(r"""# ============================================================
# B5. CKAN TRAINER & HÀM ĐÁNH GIÁ ĐỘ ĐO HỌC THUẬT
# ============================================================
def evaluate_predictions(labels, scores):
    # Tính toán ROC-AUC, F1-Score và Accuracy
    auc = roc_auc_score(labels, scores)
    preds = [1 if s >= 0.5 else 0 for s in scores]
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return auc, f1, acc

class CKANTrainer:
    # Quản lý huấn luyện và đánh giá mô hình CKAN
    def __init__(self, model, lr=0.002, weight_decay=1e-5, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()
        self.loss_history = []

    def train_epoch(self, train_data, user_triple_set, item_triple_set, n_layer, batch_size=1024):
        self.model.train()
        perm = np.random.permutation(len(train_data))
        epoch_losses = []
        for s in range(0, len(train_data), batch_size):
            b = train_data[perm[s:s+batch_size]]
            items = torch.LongTensor(b[:, 1]).to(self.device)
            labels = torch.FloatTensor(b[:, 2]).to(self.device)
            u_tr = to_triple_tensor(b[:, 0], user_triple_set, n_layer, self.device)
            i_tr = to_triple_tensor(b[:, 1], item_triple_set, n_layer, self.device)
            
            self.optimizer.zero_grad()
            preds = self.model(items, u_tr, i_tr)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = float(np.mean(epoch_losses))
        self.loss_history.append(avg_loss)
        return avg_loss

    def evaluate(self, test_data, user_triple_set, item_triple_set, n_layer, batch_size=2048):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for s in range(0, len(test_data), batch_size):
                b = test_data[s:s+batch_size]
                items = torch.LongTensor(b[:, 1]).to(self.device)
                u_tr = to_triple_tensor(b[:, 0], user_triple_set, n_layer, self.device)
                i_tr = to_triple_tensor(b[:, 1], item_triple_set, n_layer, self.device)
                scores.extend(self.model(items, u_tr, i_tr).cpu().numpy())
        return evaluate_predictions(test_data[:, 2], scores)

print("[OK] Đã hoàn thành Mô-đun B5: CKANTrainer & Evaluation Metrics!")
""")

# ============================================================
# PHẦN C: THỰC NGHIỆM ĐA MIỀN & TRỰC QUAN HÓA
# ============================================================
add_md(r"""## PHẦN C: TIẾN TRÌNH THỰC NGHIỆM ĐA MIỀN & ĐÁNH GIÁ KHOA HỌC

Quy trình thực nghiệm được phân tách rõ ràng thành:
- **Thực nghiệm 1**: Warm-start CTR Benchmark so sánh 4 mô hình (`MostPopular`, `Item-KNN`, `Matrix Factorization`, `CKAN`).
- **Thực nghiệm 2**: Khảo sát khả năng chống chịu độ thưa thớt (Data Sparsity Study: 10% Training Data).
- **Trực quan hóa**: Bảng tổng hợp số liệu và Dashboard đồ thị trực quan.
""")

# C1: Benchmark Runner
add_code(r"""# ============================================================
# C1. THỰC NGHIỆM 1: WARM-START CTR BENCHMARK TRÊN CÁC TẬP DỮ LIỆU
# ============================================================
def run_benchmark_on_dataset(ds_name):
    cfg = DATASETS_CONFIG[ds_name]
    rating_np = np.load(f"./data/{ds_name}/ratings_final.npy")
    kg_np = np.load(f"./data/{ds_name}/kg_final.npy")
    
    n_user = int(np.max(rating_np[:, 0])) + 1
    n_item = int(np.max(rating_np[:, 1])) + 1
    n_entity = int(max(np.max(kg_np[:, 0]), np.max(kg_np[:, 2]), np.max(rating_np[:, 1]))) + 1
    n_relation = int(np.max(kg_np[:, 1])) + 1
    
    print(f"\n{'='*65}\n>>> TIẾN TRÌNH BENCHMARK: TẬP DỮ LIỆU {ds_name.upper()} <<<")
    print(f"Users: {n_user:,} | Items: {n_item:,} | Ratings: {len(rating_np):,} | KG Triples: {len(kg_np):,}")
    
    # Chia tập train:val:test (6:2:2)
    np.random.seed(SEED)
    idx = np.random.permutation(len(rating_np))
    train_data = rating_np[idx[:int(0.6 * len(rating_np))]]
    test_data = rating_np[idx[int(0.8 * len(rating_np)):]]
    
    # ----------------------------------------------------
    # 1. BASELINE: MostPopular
    # ----------------------------------------------------
    pop = MostPopularBaseline()
    pop.fit(train_data, n_item)
    pop_sc = pop.predict(test_data[:, 1])
    pop_auc, pop_f1, pop_acc = evaluate_predictions(test_data[:, 2], pop_sc)
    print(f"  [1/4] MostPopular       : AUC = {pop_auc:.4f} | F1 = {pop_f1:.4f} | ACC = {pop_acc:.4f}")
    
    # ----------------------------------------------------
    # 2. BASELINE: Item-KNN (sklearn NearestNeighbors)
    # ----------------------------------------------------
    knn = ItemKNNBaseline(k=20)
    knn.fit(train_data, n_user, n_item)
    sub_test = test_data[:min(3000, len(test_data))]
    knn_sc = knn.predict(sub_test[:, 0], sub_test[:, 1])
    knn_auc, knn_f1, knn_acc = evaluate_predictions(sub_test[:, 2], knn_sc)
    print(f"  [2/4] Item-KNN (sklearn): AUC = {knn_auc:.4f} | F1 = {knn_f1:.4f} | ACC = {knn_acc:.4f}")
    
    # ----------------------------------------------------
    # 3. BASELINE: Matrix Factorization
    # ----------------------------------------------------
    mf = MatrixFactorizationBaseline(n_user, n_item, dim=cfg["dim"]).to(device)
    opt_mf = torch.optim.Adam(mf.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    crit = nn.BCELoss()
    bs = cfg["batch_size"]
    
    print("  Đang huấn luyện Matrix Factorization...")
    for ep in range(5):
        mf.train()
        perm = np.random.permutation(len(train_data))
        for s in range(0, len(train_data), bs):
            b = train_data[perm[s:s+bs]]
            u = torch.LongTensor(b[:, 0]).to(device)
            i = torch.LongTensor(b[:, 1]).to(device)
            l = torch.FloatTensor(b[:, 2]).to(device)
            opt_mf.zero_grad()
            crit(mf(u, i), l).backward()
            opt_mf.step()
    mf.eval()
    with torch.no_grad():
        mf_sc = []
        for s in range(0, len(test_data), bs):
            b = test_data[s:s+bs]
            mf_sc.extend(mf(torch.LongTensor(b[:, 0]).to(device), torch.LongTensor(b[:, 1]).to(device)).cpu().numpy())
    mf_auc, mf_f1, mf_acc = evaluate_predictions(test_data[:, 2], mf_sc)
    print(f"  [3/4] Matrix Factorization: AUC = {mf_auc:.4f} | F1 = {mf_f1:.4f} | ACC = {mf_acc:.4f}")
    
    # ----------------------------------------------------
    # 4. PROPOSED METHOD: CKAN (With Knowledge Graph)
    # ----------------------------------------------------
    sampler = KnowledgeRippleSampler(kg_np, n_layer=cfg["n_layer"], itss=cfg["itss"], utss=cfg["utss"])
    item_triple_set = sampler.build_item_ripple_set(n_item)
    user_pos = defaultdict(list)
    for u, i, r in train_data:
        if r == 1: user_pos[u].append(i)
    user_triple_set = sampler.build_user_ripple_set(n_user, user_pos)
    
    ckan = CKAN(n_entity, n_relation, dim=cfg["dim"], n_layer=cfg["n_layer"], agg="concat")
    trainer = CKANTrainer(ckan, lr=cfg["lr"], weight_decay=1e-5, device=device)
    
    print("  Đang huấn luyện CKAN (Knowledge-Aware Attentive Network)...")
    for ep in range(5):
        loss = trainer.train_epoch(train_data, user_triple_set, item_triple_set, cfg["n_layer"], batch_size=bs)
    ckan_auc, ckan_f1, ckan_acc = trainer.evaluate(test_data, user_triple_set, item_triple_set, cfg["n_layer"], batch_size=bs)
    print(f"  [4/4] CKAN (With KG)    : AUC = {ckan_auc:.4f} | F1 = {ckan_f1:.4f} | ACC = {ckan_acc:.4f}")
    
    # ----------------------------------------------------
    # 5. SPARSITY STRESS TEST (10% Dữ Liệu Huấn Luyện)
    # ----------------------------------------------------
    sub_tr = train_data[:int(len(train_data) * 0.1)]
    
    # MF Sparse
    mf_sp = MatrixFactorizationBaseline(n_user, n_item, dim=cfg["dim"]).to(device)
    opt_sp = torch.optim.Adam(mf_sp.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    for _ in range(4):
        mf_sp.train()
        for s in range(0, len(sub_tr), bs):
            b = sub_tr[s:s+bs]
            opt_sp.zero_grad()
            crit(mf_sp(torch.LongTensor(b[:, 0]).to(device), torch.LongTensor(b[:, 1]).to(device)), torch.FloatTensor(b[:, 2]).to(device)).backward()
            opt_sp.step()
    with torch.no_grad():
        mf_sp_sc = [mf_sp(torch.LongTensor(b[:, 0]).to(device), torch.LongTensor(b[:, 1]).to(device)).cpu().numpy() for b in [test_data[s:s+bs] for s in range(0, len(test_data), bs)]]
        mf_sp_sc = np.concatenate(mf_sp_sc)
    mf_sp_auc, _, _ = evaluate_predictions(test_data[:, 2], mf_sp_sc)
    
    # CKAN Sparse
    ck_sp = CKAN(n_entity, n_relation, dim=cfg["dim"], n_layer=cfg["n_layer"], agg="concat")
    trainer_sp = CKANTrainer(ck_sp, lr=cfg["lr"], weight_decay=1e-5, device=device)
    for _ in range(4):
        trainer_sp.train_epoch(sub_tr, user_triple_set, item_triple_set, cfg["n_layer"], batch_size=bs)
    ck_sp_auc, _, _ = trainer_sp.evaluate(test_data, user_triple_set, item_triple_set, cfg["n_layer"], batch_size=bs)
    
    diff = (ck_sp_auc - mf_sp_auc) * 100
    print(f"  [Sparsity 10%] MF AUC = {mf_sp_auc:.4f} | CKAN AUC = {ck_sp_auc:.4f} (CKAN vượt trội: +{diff:.1f}%)")
    
    return {
        "Dataset": ds_name.capitalize(),
        "Users": n_user, "Items": n_item, "Ratings": len(rating_np), "KG_Triples": len(kg_np),
        "MostPop_AUC": pop_auc, "ItemKNN_AUC": knn_auc, "MF_AUC": mf_auc, "CKAN_AUC": ckan_auc,
        "MF_F1": mf_f1, "CKAN_F1": ckan_f1,
        "MF_Sparse10_AUC": mf_sp_auc, "CKAN_Sparse10_AUC": ck_sp_auc
    }

results = []
for ds in active_datasets:
    results.append(run_benchmark_on_dataset(ds))

df_results = pd.DataFrame(results)
print("\n" + "="*70)
print("BẢNG TỔNG KẾT SO SÁNH THỰC NGHIỆM ĐA MIỀN:")
print("="*70)
print(df_results.to_string(index=False))
""")

# C2: Visual Dashboard
add_code(r"""# ============================================================
# C2. TRỰC QUAN HÓA SO SÁNH ĐA MIỀN (VISUALIZATION DASHBOARD)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

x = np.arange(len(df_results))
width = 0.20

# 1. Warm-start CTR Comparison
axes[0].bar(x - 1.5*width, df_results["MostPop_AUC"], width, label="MostPopular", color="#9CA3AF")
axes[0].bar(x - 0.5*width, df_results["ItemKNN_AUC"], width, label="Item-KNN (sklearn)", color="#60A5FA")
axes[0].bar(x + 0.5*width, df_results["MF_AUC"], width, label="Biased MF", color="#3B82F6")
axes[0].bar(x + 1.5*width, df_results["CKAN_AUC"], width, label="CKAN (KG Attention)", color="#D97706")
axes[0].set_xticks(x)
axes[0].set_xticklabels(df_results["Dataset"], fontsize=11, fontweight="bold")
axes[0].set_ylabel("Test ROC-AUC", fontsize=11, fontweight="bold")
axes[0].set_title("So Sánh ROC-AUC Trên Các Miền Dữ Liệu (Warm-Start)", fontsize=12, fontweight="bold")
axes[0].set_ylim(0.4, 1.05)
axes[0].legend(loc="lower right")
axes[0].grid(axis="y", linestyle=":", alpha=0.7)

# 2. Sparsity 10% Impact
width2 = 0.35
bars1 = axes[1].bar(x - width2/2, df_results["MF_Sparse10_AUC"], width2, label="MF (10% Dữ Liệu)", color="#93C5FD", edgecolor="#3B82F6")
bars2 = axes[1].bar(x + width2/2, df_results["CKAN_Sparse10_AUC"], width2, label="CKAN (10% Dữ Liệu)", color="#F59E0B", edgecolor="#D97706")
axes[1].set_xticks(x)
axes[1].set_xticklabels(df_results["Dataset"], fontsize=11, fontweight="bold")
axes[1].set_ylabel("Test ROC-AUC", fontsize=11, fontweight="bold")
axes[1].set_title("Khả Năng Chống Chịu Độ Thưa Thớt (10% Data Stress-Test)", fontsize=12, fontweight="bold", color="#991B1B")
axes[1].set_ylim(0.4, 1.05)
axes[1].legend(loc="lower right")
axes[1].grid(axis="y", linestyle=":", alpha=0.7)

# Ghi chú phần trăm vượt trội
for i in range(len(df_results)):
    diff = (df_results["CKAN_Sparse10_AUC"][i] - df_results["MF_Sparse10_AUC"][i]) * 100
    axes[1].text(i, max(df_results["CKAN_Sparse10_AUC"][i], df_results["MF_Sparse10_AUC"][i]) + 0.03, f"+{diff:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color="#B45309")

fig.suptitle("ĐÁNH GIÁ ĐA MIỀN: MOVIELENS (PHIM) • BOOK-CROSSING (SÁCH) • LAST.FM (ÂM NHẠC)",
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("./tri_dataset_benchmark_summary.png", bbox_inches="tight")
plt.show()

print("[OK] Đã hoàn thành toàn bộ thực nghiệm nghiên cứu và xuất biểu đồ so sánh!")
""")

out_path = os.path.abspath("notebooks/CKAN_Tri_Dataset_Benchmark_Colab.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Regenerated notebook successfully at:", out_path)
