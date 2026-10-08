import sys
import json

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("=" * 70)
print(">>> 1. KIỂM TRA HEALTH CHECK BACKEND <<<")
print("=" * 70)
res = client.get("/health")
print("HTTP Status:", res.status_code)
print("Response:", json.dumps(res.json(), indent=2, ensure_ascii=False))
assert res.status_code == 200

print("\n" + "=" * 70)
print(">>> 2. KIỂM TRA DANH SÁCH 3 MIỀN DỮ LIỆU (/api/v1/domains) <<<")
print("=" * 70)
res = client.get("/api/v1/domains")
assert res.status_code == 200
domains = res.json()["domains"]
for d in domains:
    print(f"[{d['id'].upper()}] - {d['name']} ({d['vietnameseName']})")
    print(f"   Items: {d['itemsCount']:,} | Triples: {d['triplesCount']:,} | Quan hệ: {d['relationsCount']}")
    print(f"   Sample demo users: {d['sampleUsers']}")

print("\n" + "=" * 70)
print(">>> 3. BÀI TOÁN 1: GỢI Ý TOP-K (3 DATASETS) <<<")
print("=" * 70)
for d in ["movie", "book", "music"]:
    uid = 1 if d == "movie" else (790 if d == "book" else 774)
    res = client.get(f"/api/v1/recommendations?domain={d}&userId={uid}&topK=5")
    assert res.status_code == 200, f"Failed {d}: {res.text}"
    data = res.json()
    print(f"\nMiền [{d.upper()}] (Người dùng #{uid}) -> Trả về {data['total']} đề xuất:")
    for item in data["recommendations"]:
        sub = item.get("subtitle") or ""
        sec = item.get("secondaryInfo") or ""
        extra = f" | {sub} - {sec}" if sub or sec else ""
        print(f"   • [{item['score']*100:.1f}%] #{item['id']} {item['title']}{extra}")
        if item.get("reasons"):
            print(f"     -> Lý do KG: {item['reasons'][0]}")

print("\n" + "=" * 70)
print(">>> 4. BÀI TOÁN 2: SUY LUẬN ĐƯỜNG DẪN TRI THỨC (/api/v1/explainability) <<<")
print("=" * 70)
test_items = {"movie": 234, "book": 10, "music": 67}
for d, item_id in test_items.items():
    uid = 1 if d == "movie" else (790 if d == "book" else 774)
    res = client.get(f"/api/v1/explainability/{item_id}?domain={d}&userId={uid}")
    assert res.status_code == 200, f"Failed {d} explain: {res.text}"
    exp = res.json()
    print(f"\nMiền [{d.upper()}] -> Mục #{item_id}:")
    print(f"   Độ tin cậy: {exp['confidence']} | Điểm: {exp['score']}")
    print(f"   Tóm tắt: {exp['executiveSummary']}")
    print(f"   Phản thực tế: {exp['counterfactual']}")
    print(f"   Số đường dẫn 2-chặng tìm thấy: {len(exp['paths'])}")
    if exp["paths"]:
        for p in exp["paths"][:2]:
            print(f"     - {p['naturalLanguage']}")
    print(f"   Phân bổ trọng số đặc trưng: {exp['featureImportance']}")
    if exp.get("subgraph"):
        print(f"   Subgraph: {len(exp['subgraph']['nodes'])} nodes, {len(exp['subgraph']['edges'])} edges")

print("\n" + "=" * 70)
print(">>> 5. BÀI TOÁN 3: THỬ NGHIỆM ĐỘ THƯA & KHỞI ĐỘNG LẠNH (/api/v1/benchmark/cold-start) <<<")
print("=" * 70)
for d in ["movie", "book", "music"]:
    res = client.get(f"/api/v1/benchmark/cold-start?domain={d}&interactions=3")
    assert res.status_code == 200
    b = res.json()
    m = b["currentMetrics"]
    print(f"\nMiền [{d.upper()}] tại mức N=3 tương tác:")
    print(f"   Mô tả: {b['description']}")
    print(f"   ROC-AUC     : CF Baseline = {m['cf_auc']:.4f} vs CKAN = {m['ckan_auc']:.4f} (Chênh lệch: +{m['delta_auc_pct']}%)")
    print(f"   Recall@10   : CF Baseline = {m['cf_recall10']:.4f} vs CKAN = {m['ckan_recall10']:.4f} (Chênh lệch: +{m['delta_recall_pct']}%)")
    print(f"   F1-Score    : CF Baseline = {m['cf_f1']:.4f} vs CKAN = {m['ckan_f1']:.4f}")
    print(f"   NDCG@10     : CF Baseline = {m['cf_ndcg10']:.4f} vs CKAN = {m['ckan_ndcg10']:.4f}")
    print(f"   Số điểm trên đường cong suy giảm: {len(b['trajectory'])}")
    print(f"   CF gợi ý mẫu #{b['cfRecommendations'][0]['id']}: {b['cfRecommendations'][0]['title']}")
    print(f"   CKAN gợi ý mẫu #{b['ckanRecommendations'][0]['id']}: {b['ckanRecommendations'][0]['title']}")

print("\n" + "=" * 70)
print(">>> 6. TƯƠNG TÁC THỜI GIAN THỰC (FEEDBACK LOOP) <<<")
print("=" * 70)
for d in ["movie", "book", "music"]:
    res = client.post("/api/v1/recommendations/feedback", json={
        "domain": d,
        "userId": 999,
        "itemId": 5,
        "action": "LIKE"
    })
    assert res.status_code == 200
    print(f"Feedback [{d.upper()}]:", res.json()["message"])

print("\n" + "=" * 70)
print(">>> 7. TRA CỨU DANH MỤC SẢN PHẨM (/api/v1/items) <<<")
print("=" * 70)
queries = [("movie", "matrix"), ("book", "ring"), ("music", "beatles")]
for d, q in queries:
    res = client.get(f"/api/v1/items?domain={d}&page=1&limit=3&search={q}")
    assert res.status_code == 200
    cat = res.json()
    print(f"\nTìm kiếm [{d.upper()}] với từ khóa '{q}': Tìm thấy {cat['total']} kết quả")
    for item in cat["data"]:
        print(f"   • #{item['id']}: {item['title']} ({item.get('subtitle') or ''})")

print("\n" + "=" * 70)
print(">>> TẤT CẢ CÁC BÀI TOÁN & DATASET ĐÃ ĐƯỢC XÁC THỰC THÀNH CÔNG 100%! <<<")
print("=" * 70)
