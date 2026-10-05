import json
import os

nb_path = os.path.abspath("notebooks/CKAN_Tri_Dataset_Benchmark_Colab.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Insert comprehensive methodology cell right after the header cell
methodology_md = """## 2. PHƯƠNG PHÁP LUẬN & KIẾN TRÚC MÔ HÌNH CKAN (COLLABORATIVE KNOWLEDGE-AWARE ATTENTIVE NETWORK)

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
   - Chiến lược Ghép nối (Concat - Khuyên dùng):
     $$\mathbf{e}_u = [\mathbf{e}_u^{(L)} \parallel \dots \parallel \mathbf{e}_u^{(0)}], \quad \mathbf{e}_v = [\mathbf{e}_v^{(L)} \parallel \dots \parallel \mathbf{e}_v^{(0)}]$$
   - Giúp bảo toàn không gian đặc trưng giữa thực thể gốc và tri thức lan truyền.

5. **Dự Đoán Tương Tác & Tối Ưu Hóa Hàm Mất Mát**:
   - Xác suất tương tác dự đoán:
     $$\hat{y}(u, v) = \sigma(\mathbf{e}_u^T \mathbf{e}_v) = \frac{1}{1 + \exp(-\mathbf{e}_u^T \mathbf{e}_v)}$$
   - Tối ưu hóa bằng Binary Cross-Entropy Loss kết hợp phạt điều chuẩn $L_2$ Weight Decay:
     $$\mathcal{L} = -\sum_{(u, v) \in \mathcal{D}} \left[ y_{u, v} \log \hat{y}(u, v) + (1 - y_{u, v}) \log(1 - \hat{y}(u, v)) \right] + \lambda \|\Theta\|_2^2$$

---

### 2.3. So Sánh Tính Chất Của 3 Tập Dữ Liệu Nghiên Cứu

| Chỉ số thống kê | MovieLens-1M (`movie`) | Book-Crossing (`book`) | Last.FM (`music`) |
| :--- | :---: | :---: | :---: |
| **Miền ứng dụng** | Điện ảnh (Phim) | Sách & Văn học | Âm nhạc (Nghệ sĩ) |
| **Số Người dùng (Users)** | 2,500 | 17,860 | 1,872 |
| **Số Sản phẩm (Items)** | 16,946 | 14,967 | 3,846 |
| **Số Lượng Tương tác** | 238,442 | 139,746 | 42,346 |
| **Độ Thưa Thớt (Sparsity)** | 99.44% | **> 99.97% (Cực thưa)** | 99.41% |
| **Số Thực Thể KG (Entities)** | 102,569 | 77,903 | 9,366 |
| **Số Quan Hệ KG (Relations)**| 32 | 25 | **60 (Đa dạng nhất)** |
| **Tổng Số Bộ Ba (Triples)** | 499,474 | 151,500 | 15,518 |
"""

new_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [l + '\n' for l in methodology_md.split('\n')]
}

# Insert at index 1
nb['cells'].insert(1, new_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Updated CKAN_Tri_Dataset_Benchmark_Colab.ipynb with comprehensive methodology!")
