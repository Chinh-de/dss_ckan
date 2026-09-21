# DSS CKAN Movie Recommender System


## Kiến Trúc Hệ Thống

```
                                  Client Browser
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   Frontend (React 18 + Vite)  │
                        │   Tailwind + Shadcn + pnpm    │
                        │        Port: 5173 / 80        │
                        └───────────────┬───────────────┘
                                        │ REST API
                                        ▼
                        ┌───────────────────────────────┐
                        │   Backend (Python FastAPI)    │
                        │   PyTorch CKAN + uv Manager   │
                        │          Port: 8000           │
                        └───────┬───────────────┬───────┘
                                │               │
                ┌───────────────┘               └───────────────┐
                ▼                                               ▼
     ┌─────────────────────┐                         ┌─────────────────────┐
     │  PostgreSQL 16 DB   │                         │  Neo4j 5 Graph DB   │
     │  (SQLAlchemy ORM)   │                         │  (104,430 entities) │
     │     Port: 5435      │                         │  Ports: 7474 / 7687 │
     └─────────────────────┘                         └─────────────────────┘
```

---

## Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### Yêu Cầu Môi Trường

* **Python**: $\ge 3.11$ kèm công cụ **`uv`** (`pip install uv` hoặc `winget install astral-sh.uv`)
* **Node.js**: $\ge 20.0$ kèm công cụ **`pnpm`** (`npm install -g pnpm`)
* **Docker & Docker Compose** (để chạy PostgreSQL và Neo4j)

---

### Thiết Lập Biến Môi Trường (.env)

Dự án đã chuẩn bị sẵn các tệp mẫu `.env.example` và tệp cấu hình mặc định sẵn sàng chạy:

* **Root**: `.env.example` và `.env` (chứa cổng cấu hình cho docker-compose).
* **Backend**: `backend/.env.example` và `backend/.env` (cấu hình DB, Neo4j, JWT, đường dẫn Data/Models).
* **Frontend**: `frontend/.env.example` và `frontend/.env` (cấu hình API endpoint).

Khi cần thay đổi mật khẩu hoặc cổng, chỉ cần copy từ `.env.example` sang `.env`:

```bash
# Ở thư mục gốc hoặc trong backend/frontend:
cp .env.example .env
```

---

### Cách 1: Khởi Chạy Bằng Docker Compose (Nhanh Nhất & Khuyên Dùng)

Chỉ với **1 câu lệnh duy nhất**, toàn bộ 4 container (`postgres`, `neo4j`, `backend`, `frontend`) sẽ được biên dịch và khởi chạy tự động:

```bash
docker compose up -d --build
```

Sau khi khởi động thành công:

* **Giao diện Web**: [http://localhost:5173/](http://localhost:5173/)
* **FastAPI Swagger API**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Neo4j Browser**: [http://localhost:7474/](http://localhost:7474/) (User: `neo4j`, Pass: `neo4j_password`)
* **PostgreSQL**: `localhost:5435` (DB: `ckan_recommendation`, User: `postgres`, Pass: `postgres_password`)

---

### Cách 2: Khởi Chạy Từng Phần Cho Phát Triển (Local Development)

#### Bước 1: Khởi động CSDL PostgreSQL & Neo4j

```bash
docker compose up -d postgres neo4j
```

#### Bước 2: Nạp Dữ Liệu Khởi Tạo (Seed PostgreSQL & Neo4j)

Chạy lệnh seeder hợp nhất để nạp **toàn bộ 16,954 phim**, **tất cả 2,500 người dùng benchmark**, **238,442 tương tác LIKE/DISLIKE**, cùng toàn bộ liên kết Đồ thị Tri thức:

```bash
cd backend

# Nạp toàn bộ 2,500 users & toàn bộ ratings (Mặc định):
uv run python seed.py

# Hoặc chỉ nạp nhanh 100 users nếu muốn test nhẹ:
uv run python seed.py --max-users 100
```

*(Tùy chọn: Thêm cờ `--clean` để xóa sạch làm mới, hoặc `--postgres-only` / `--neo4j-only` khi cần).*

#### Bước 3: Khởi động Backend với `uv`

```bash
# Ở thư mục backend:
uv sync
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

* Kiểm tra API Health: [http://localhost:8000/health](http://localhost:8000/health)
* Tài liệu Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

#### Bước 4: Khởi động Frontend với `pnpm`

Mở một terminal mới:

```bash
cd frontend

# Cài đặt thư viện bằng pnpm
pnpm install

# Khởi chạy Vite dev server
pnpm dev
```

* Truy cập Web: [http://localhost:5173/](http://localhost:5173/)

---

## Kiểm Thử Tự Động (Automated Testing)

### 1. Kiểm thử Backend (Pytest qua `uv`)

```bash
cd backend
uv run --extra dev pytest tests/
```

* Kết quả: Xác nhận hàm sinh động `user_triple_set` chạy thành công trong $< 5\text{ ms}$.

### 2. Kiểm thử Build Frontend (TypeScript & CSS qua `pnpm`)

```bash
cd frontend
pnpm build
```

* Kết quả: Biên dịch toàn bộ production bundle chỉ trong $\approx 4.9\text{s}$ với 0 lỗi TypeScript, 0 lỗi CSS.

---

## Danh Mục REST API Chính

| Phương thức | Endpoint                                 | Mô tả                                                                              |
| :------------- | :--------------------------------------- | :----------------------------------------------------------------------------------- |
| `POST`       | `/api/v1/auth/register`                | Đăng ký tài khoản người dùng                                                 |
| `POST`       | `/api/v1/auth/login`                   | Đăng nhập và nhận JWT token                                                     |
| `POST`       | `/api/v1/auth/onboarding`              | Khởi tạo gu thể loại và phim khởi đầu cho user mới                          |
| `GET`        | `/api/v1/movies`                       | Danh sách phim (phân trang, tìm kiếm theo tên, lọc thể loại)                 |
| `GET`        | `/api/v1/movies/{id}`                  | Chi tiết phim và điểm đánh giá trung bình                                    |
| `GET`        | `/api/v1/movies/{id}/related`          | Lấy danh sách phim tương tự từ đồ thị Neo4j                                 |
| `POST`       | `/api/v1/ratings`                      | Đánh giá phim 1-5 sao, gắn nhãn LIKE/DISLIKE, sync Neo4j tức thì              |
| `GET`        | `/api/v1/recommendations`              | **Gợi ý Top-K cá nhân hóa** (hỗ trợ user mới, tự động lọc dislike) |
| `GET`        | `/api/v1/explainability/{movieId}`     | Trích xuất đường dẫn giải thích đa tầng Cypher                             |
| `GET`        | `/api/v1/graph/subgraph/{movieId}`     | Subgraph 1-hop quanh phim phục vụ biểu đồ                                       |
| `GET`        | `/api/v1/graph/user-subgraph/{userId}` | Subgraph kết nối User đến các phim gợi ý qua thực thể                       |

---

## Cấu Trúc Mã Nguồn

```
dss_ckan_movie_recommender_system/
├── backend/                  # FastAPI Backend quản lý bằng uv
│   ├── app/
│   │   ├── api/v1/           # Các REST routers
│   │   ├── core/             # Config, Database, Neo4j, Security
│   │   ├── models/           # SQLAlchemy models & Pydantic schemas
│   │   ├── recommendation/   # CKAN PyTorch, dynamic propagation, engine
│   │   └── main.py           # FastAPI entrypoint
│   ├── data/movie/           # Metadata, Knowledge Graph & tương tác (tự chứa 100%)
│   ├── models/               # Weights checkpoint (*.pt) & cache
│   ├── tests/                # Pytest unit tests
│   ├── pyproject.toml        # Cấu hình dự án uv
│   ├── requirements.txt      # Dependencies
│   ├── .env.example          # Mẫu biến môi trường backend
│   ├── .env                  # Cấu hình biến môi trường backend
│   ├── .gitignore            # Gitignore riêng cho backend
│   └── Dockerfile            # Container build với base uv
├── frontend/                 # React Vite Frontend quản lý bằng pnpm
│   ├── src/
│   │   ├── components/       # UI, Navbar, MovieCard, Onboarding, Modals
│   │   ├── services/         # API client
│   │   ├── types/            # TypeScript interfaces
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json          # Dependencies pnpm
│   ├── tailwind.config.js    # Design tokens Cinematic Luxury
│   ├── .env.example          # Mẫu biến môi trường frontend
│   ├── .env                  # Cấu hình biến môi trường frontend
│   ├── .gitignore            # Gitignore riêng cho frontend
│   └── Dockerfile            # Nginx production container
├── .agent/skills/            # 3 Design Skills (uiuxpromax, taste-skill, claude-design)
├── docker-compose.yml        # Orchestration đồng bộ 4 container
├── .env.example              # Mẫu biến môi trường root
├── .env                      # Cấu hình biến môi trường root
├── .gitignore                # Bỏ qua venv, node_modules, weights lớn
└── README.md                 # Tài liệu hướng dẫn chi tiết
```
