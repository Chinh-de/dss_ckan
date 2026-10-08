# Đồ thị tri thức cho hệ gợi ý: RippleNet và CKAN

Đề tài cài đặt lại RippleNet (CIKM 2018) và CKAN (SIGIR 2020), so sánh với MostPopular và Matrix Factorization trên ba tập dữ liệu (MovieLens-20M, Book-Crossing, Last.FM), và dựng một ứng dụng web minh hoạ dùng mô hình CKAN đã huấn luyện.

Kho mã gồm ba phần:

| Phần | Vị trí | Nội dung |
| :--- | :--- | :--- |
| Thực nghiệm | `notebooks/` | Notebook huấn luyện và đánh giá bốn mô hình |
| Ứng dụng minh hoạ | `backend/`, `frontend/` | API FastAPI + giao diện React |
| Báo cáo | `docs/` | Báo cáo `.docx`/`.pdf` và mã sinh báo cáo |

```
Trình duyệt ──► Frontend (React 18 + Vite, cổng 5173)
                    │ REST /api/v1
                    ▼
                Backend (FastAPI + PyTorch CKAN, cổng 8000)
                    ├── saved_models/   mô hình do notebook xuất
                    ├── backend/data/   dữ liệu, đồ thị tri thức, metadata của ba tập
                    ├── PostgreSQL 16   người dùng và tương tác của tập phim (cổng 5435)
                    └── Neo4j 5         đồ thị của tập phim (cổng 7474 / 7687)
```

---

## 1. Chạy thực nghiệm (notebook)

Notebook chính là `notebooks/dss-knowledgegraph-for-rs-fixed.ipynb` (bản chưa chạy). Bản đã chạy trên GPU Tesla T4, còn nguyên kết quả, là `notebooks/dss-knowledgegraph-for-rs (3).ipynb`.

1. Tải `dss-knowledgegraph-for-rs-fixed.ipynb` lên Kaggle hoặc Google Colab, bật GPU.
2. Chạy toàn bộ các ô từ trên xuống. Notebook tự tải dữ liệu, huấn luyện MostPopular, MF, RippleNet, CKAN trên ba tập và in ba bảng kết quả (CTR, Recall@K, độ thưa).
3. Ô cuối xuất mô hình CKAN vào `./saved_models/<movie|book|music>/` và nén thành `ckan_tri_dataset_models.zip`. Tải file này về.

Notebook được sinh từ `notebooks/_build_fixed_notebook.py`. Khi cần sửa mã notebook, sửa trong file đó rồi chạy lại:

```bash
python notebooks/_build_fixed_notebook.py
```

### Đưa kết quả notebook vào ứng dụng

```bash
# 1. Giải nén mô hình vào thư mục saved_models/ ở gốc kho mã, sao cho có:
#    saved_models/movie/ckan_model.pt, config.json, item_embeddings.npy  (tương tự cho book, music)

# 2. Đặt notebook đã chạy vào notebooks/ rồi trích số đo sang backend/data/benchmark_results.json
cd backend
uv run python -m app.scripts.extract_benchmark_results
```

Tên notebook được đọc nằm ở biến `NOTEBOOK` trong `backend/app/scripts/extract_benchmark_results.py`; đổi biến này nếu file có tên khác.

`saved_models/` không nằm trong git vì quá lớn (khoảng 210 MB), nên sau khi clone cần làm bước 1 ở trên để ứng dụng có mô hình của đủ ba tập. Khi thiếu thư mục này, backend tìm checkpoint dự phòng ở `backend/models/ckan_model.pt` (phim) và `backend/data/<book|music>/ckan_checkpoint.pt`; kho mã chỉ kèm sẵn checkpoint của tập nhạc.

---

## 2. Chạy ứng dụng minh hoạ

### Yêu cầu

* Python ≥ 3.11 và [`uv`](https://docs.astral.sh/uv/) (`pip install uv`)
* Node.js ≥ 20 và `pnpm` (`npm install -g pnpm`)
* Docker và Docker Compose (cho PostgreSQL và Neo4j)

### Biến môi trường

Mỗi thư mục có sẵn file mẫu; sao chép thành `.env` và sửa nếu cần đổi cổng hoặc mật khẩu:

```bash
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

Hai biến đáng chú ý của backend: `CORS_ORIGINS` (các địa chỉ giao diện được phép gọi API) và `SAVED_MODELS_DIR` (thư mục mô hình do notebook xuất, mặc định `../saved_models`).

### Cách 1: Docker Compose

```bash
docker compose up -d --build
```

### Cách 2: Chạy từng phần

```bash
# Bước 1: cơ sở dữ liệu
docker compose up -d postgres neo4j

# Bước 2: nạp dữ liệu tập phim vào PostgreSQL và Neo4j (chỉ cần làm một lần)
cd backend
uv sync
uv run python seed.py                 # thêm --max-users 100 để nạp nhanh, --clean để nạp lại từ đầu

# Bước 3: backend
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

# Bước 4: frontend (terminal khác)
cd frontend
pnpm install
pnpm dev
```

Backend nạp mô hình của cả ba tập khi khởi động nên cần vài chục giây trước khi nhận yêu cầu.

### Địa chỉ

* Giao diện: http://localhost:5173
* Tài liệu API (Swagger): http://localhost:8000/docs
* Neo4j Browser: http://localhost:7474 (`neo4j` / `neo4j_password`)
* PostgreSQL: `localhost:5435`, cơ sở dữ liệu `ckan_recommendation` (`postgres` / `postgres_password`)

### Các màn hình

Thanh trên cùng cho phép đổi tập dữ liệu (phim, sách, nhạc) và đổi hoặc tạo người dùng.

* **Gợi ý Top-K**: danh sách gợi ý của CKAN; bấm vào một gợi ý để xem đường dẫn tri thức lý giải.
* **Đồ thị tri thức**: đồ thị nối người dùng, các mục đã thích, thực thể chung và các mục được gợi ý.
* **Độ thưa dữ liệu**: số đo của notebook theo lượng dữ liệu huấn luyện và bảng kết quả bốn mô hình.
* **Thư viện**: duyệt, tìm kiếm và thích các mục để thay đổi gợi ý.

---

## 3. Dữ liệu bổ trợ (tuỳ chọn)

Các file dưới đây đã có sẵn trong kho mã; chỉ cần chạy lại khi muốn làm mới. Cả hai lệnh đều cần mạng và không cần khoá API.

```bash
cd backend
uv run python -m app.scripts.build_entity_names      # tên thực thể của đồ thị phim và sách (Wikidata)
uv run python -m app.scripts.fetch_artist_images     # ảnh nghệ sĩ của tập nhạc (Deezer)
```

---

## 4. Kiểm thử

```bash
cd backend
uv run --extra dev pytest tests/

cd frontend
pnpm build          # kiểm tra kiểu TypeScript và dựng bản production
```

---

## 5. Dựng lại báo cáo

Báo cáo nằm ở `docs/Bao_cao_KG4RS_RippleNet_CKAN.docx` (và `.pdf`). Mọi số liệu trong báo cáo được đọc từ `backend/data/benchmark_results.json`, nên sau khi chạy lại notebook chỉ cần trích số (mục 1) rồi dựng lại:

```bash
# Cần backend chạy ở cổng 8000; cài thêm: pip install python-docx matplotlib pillow
python docs/report/make_figures.py        # biểu đồ, sơ đồ, công thức -> docs/report/figures, docs/report/eq
python docs/report/build_report.py        # sinh file .docx
pwsh docs/report/finalize.ps1             # cập nhật mục lục và xuất PDF (cần Microsoft Word)
```

Ảnh chụp màn hình của Chương 7 nằm ở `docs/report/screens/` (không nằm trong git). Để chụp lại, chạy backend và frontend rồi:

```bash
node docs/report/capture_screens.mjs docs/report/screens <thư-mục-tạm-cho-hồ-sơ-trình-duyệt>
```

Lệnh này dùng Microsoft Edge ở chế độ không giao diện.

---

## 6. Cấu trúc thư mục

```
├── notebooks/
│   ├── dss-knowledgegraph-for-rs-fixed.ipynb    # notebook thực nghiệm (chưa chạy)
│   ├── dss-knowledgegraph-for-rs (3).ipynb      # bản đã chạy, nguồn số liệu của báo cáo
│   └── _build_fixed_notebook.py                 # mã sinh notebook
├── backend/
│   ├── app/
│   │   ├── api/v1/           # các router REST
│   │   ├── core/             # cấu hình, PostgreSQL, Neo4j, bảo mật
│   │   ├── models/           # SQLAlchemy models và Pydantic schemas
│   │   ├── recommendation/   # mô hình CKAN, bộ máy gợi ý và lý giải
│   │   └── scripts/          # trích số đo notebook, tên thực thể, ảnh nghệ sĩ
│   ├── data/                 # dữ liệu ba tập, benchmark_results.json
│   ├── tests/
│   └── seed.py               # nạp dữ liệu tập phim vào PostgreSQL và Neo4j
├── frontend/
│   └── src/                  # components, lib, services, types
├── docs/
│   ├── Bao_cao_KG4RS_RippleNet_CKAN.docx / .pdf
│   └── report/               # mã sinh báo cáo, hình và công thức
├── saved_models/             # mô hình do notebook xuất (không nằm trong git)
└── docker-compose.yml
```
