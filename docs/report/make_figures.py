"""Draw every chart, block diagram and equation used by the report.

Charts read the project's real artefacts only: backend/data/benchmark_results.json (numbers
copied from the executed notebook), the KG / rating arrays, and the running backend for the
explanation subgraph. Run from the repository root with the backend up on port 8000:

    backend/.venv/Scripts/python.exe docs/report/make_figures.py
"""
import collections
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
EQ = Path(__file__).resolve().parent / "eq"
OUT.mkdir(exist_ok=True)
EQ.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "backend"))

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

NAVY, BLUE, SLATE, GRAY = "#1E3A8A", "#2563EB", "#334155", "#94A3B8"
MODEL_COLOR = {"MostPopular": "#94A3B8", "MF": "#64748B", "RippleNet": "#D97706", "CKAN": "#1E3A8A"}
MODEL_STYLE = {"MostPopular": ":", "MF": "--", "RippleNet": "-.", "CKAN": "-"}
MODEL_MARKER = {"MostPopular": "v", "MF": "s", "RippleNet": "^", "CKAN": "o"}
DATASETS = [("movie", "MovieLens-20M"), ("book", "Book-Crossing"), ("music", "Last.FM")]

BENCH = json.loads((ROOT / "backend" / "data" / "benchmark_results.json").read_text(encoding="utf-8"))["datasets"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("figure", name)


# --------------------------------------------------------------------------- charts
def chart_ctr():
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1), sharey=False)
    models = ["MostPopular", "MF", "RippleNet", "CKAN"]
    x = np.arange(len(DATASETS))
    w = 0.2
    for ax, (key, title) in zip(axes, [("auc", "ROC-AUC"), ("f1", "F1-Score"), ("acc", "Accuracy")]):
        for i, m in enumerate(models):
            vals = [BENCH[d]["models"][m][key] for d, _ in DATASETS]
            bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=MODEL_COLOR[m], edgecolor="white", linewidth=0.6)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6.2, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(["Phim", "Sách", "Nhạc"], fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylim(0, 1.16)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Giá trị trên tập kiểm thử")
    axes[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9.5)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "chart_ctr")


def chart_recall():
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    ks = ["5", "10", "20", "50", "100"]
    for ax, (d, name) in zip(axes, DATASETS):
        for m in ["MostPopular", "MF", "RippleNet", "CKAN"]:
            vals = [BENCH[d]["models"][m]["recall"][k] for k in ks]
            ax.plot(range(len(ks)), vals, MODEL_STYLE[m], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], markersize=5, linewidth=1.8, label=m)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_xlabel("K")
        ax.set_title(name, fontweight="bold", fontsize=11)
    axes[0].set_ylabel("Recall@K")
    axes[1].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, fontsize=9.5)
    fig.subplots_adjust(wspace=0.36)
    save(fig, "chart_recall")


def chart_sparsity():
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    for ax, (d, name) in zip(axes, DATASETS):
        sp = BENCH[d]["sparsity"]
        labels = [f"{int(r * 100)}%" for r in sp["ratios"]]
        for m in ["MF", "RippleNet", "CKAN"]:
            ax.plot(range(len(labels)), sp["auc"][m], MODEL_STYLE[m], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], markersize=5, linewidth=1.8, label=m)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Tỷ lệ tập huấn luyện", fontsize=9.5)
        ax.set_title(name, fontweight="bold", fontsize=11)
    axes[0].set_ylabel("ROC-AUC trên tập kiểm thử")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, fontsize=9.5)
    fig.subplots_adjust(wspace=0.36)
    save(fig, "chart_sparsity")


def chart_gain():
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    colors = {"movie": "#B45309", "book": "#047857", "music": "#0369A1"}
    markers = {"movie": "o", "book": "s", "music": "^"}
    for d, name in DATASETS:
        sp = BENCH[d]["sparsity"]
        gain = [(c - m) / m * 100 for c, m in zip(sp["auc"]["CKAN"], sp["auc"]["MF"])]
        xs = range(len(gain))
        ax.plot(xs, gain, color=colors[d], marker=markers[d], linewidth=1.8, markersize=5, label=name)
        ax.annotate(f"{gain[0]:+.1f}%", (0, gain[0]), textcoords="offset points", xytext=(6, 4), fontsize=9, color=colors[d])
        ax.annotate(f"{gain[-1]:+.1f}%", (len(gain) - 1, gain[-1]), textcoords="offset points", xytext=(-8, 6), fontsize=9, color=colors[d], ha="right")
    ax.axhline(0, color=SLATE, linewidth=0.9)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{int(r * 100)}%" for r in BENCH["movie"]["sparsity"]["ratios"]])
    ax.set_xlabel("Tỷ lệ tập huấn luyện được giữ lại")
    ax.set_ylabel("Chênh lệch AUC của CKAN so với MF (%)")
    ax.legend(frameon=False, fontsize=9.5)
    save(fig, "chart_gain")


def load_domain(d):
    kg = np.load(ROOT / "backend" / "data" / d / "kg_final.npy")
    rel = json.loads((ROOT / "backend" / "data" / d / "kg_relations.json").read_text(encoding="utf-8"))["index2rel"]
    n_item = sum(1 for _ in open(ROOT / "backend" / "data" / d / "item_index2entity_id.txt", encoding="utf-8"))
    return kg, rel, n_item


def chart_relations():
    from app.recommendation.relation_labels import relation_role

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.4))
    for ax, (d, name) in zip(axes, DATASETS):
        kg, rel, _ = load_domain(d)
        counts = collections.Counter(kg[:, 1].tolist()).most_common(8)[::-1]
        labels = [f"{relation_role(rel[str(r)])}  ({'.'.join(rel[str(r)].split('.')[-2:])})" for r, _ in counts]
        vals = [c for _, c in counts]
        ax.barh(range(len(vals)), vals, color=NAVY, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:,}".replace(",", "."), va="center", fontsize=9)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(f"{name} ({len(kg):,} bộ ba)".replace(",", "."), fontweight="bold", fontsize=11)
        ax.set_xlim(0, max(vals) * 1.16)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    save(fig, "chart_relations")


def chart_kg_structure():
    """Out-degree of items, depth of the KG and how often a top-10 recommendation has a KG path."""
    from app.recommendation.engine import recommendation_engine as engine

    engine.initialize()
    names, med, deeper, cover = [], [], [], []
    degree_hist = {}
    for d, name in DATASETS:
        dom = engine.get_domain(d)
        deg = np.array([len(dom.kg_dict.get(i, [])) for i in dom.all_items])
        tails = {t for h in dom.kg_dict for t, _ in dom.kg_dict[h]}
        deep = sum(1 for t in tails if dom.kg_dict.get(t))
        rng = np.random.RandomState(1)
        users = [u for u in rng.permutation(list(dom.user_history))[:60] if dom.user_history[u]]
        hit = total = 0
        for u in users:
            liked_tails = {t for h in dom.user_history[u] for t, _ in dom.kg_dict.get(h, [])}
            for rec in engine.recommend(d, int(u), 10):
                total += 1
                hit += any(t in liked_tails for t, _ in dom.kg_dict.get(rec.id, []))
        names.append(name)
        med.append(float(np.median(deg)))
        deeper.append(deep / len(tails) * 100)
        cover.append(hit / total * 100)
        degree_hist[name] = deg
    stats = {"datasets": names, "median_triples_per_item": med, "tails_with_next_hop_pct": deeper,
             "top10_with_kg_path_pct": cover, "users_sampled": 60}
    (OUT / "kg_structure.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
    panels = [(med, "Số bộ ba của mỗi item (trung vị)", "{:.0f}"),
              (deeper, "Thực thể đuôi còn cạnh đi tiếp (%)", "{:.1f}%"),
              (cover, "Gợi ý top-10 có đường dẫn KG (%)", "{:.0f}%")]
    for ax, (vals, title, fmt) in zip(axes, panels):
        bars = ax.bar(["Phim", "Sách", "Nhạc"], vals, color=[NAVY, "#047857", "#0369A1"], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=9.6)
        ax.set_ylim(0, max(vals) * 1.2 + 1)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=9)
    fig.subplots_adjust(wspace=0.3)
    save(fig, "chart_kg_structure")


# --------------------------------------------------------------------------- diagrams
def canvas(w, h, scale=0.8):
    fig, ax = plt.subplots(figsize=(w * scale, h * scale))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc="#EFF6FF", ec=NAVY, fs=9.5, bold=False, color="#0F172A", style="round,pad=0.25,rounding_size=1.2", ls=1.25):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=color, linespacing=ls)


def elbow(ax, points, color=SLATE):
    """Polyline with an arrow head on the last segment."""
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=1.2, solid_capstyle="round")
    arrow(ax, points[-2], points[-1], color=color)


def arrow(ax, a, b, text=None, color=SLATE, rad=0.0, fs=8.5, dy=1.2, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, lw=1.2, color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=fs, color=color, style="italic")


LIGHT, GREEN, AMBER, ROSE, WHITE = "#EFF6FF", "#ECFDF5", "#FFFBEB", "#FEF2F2", "#FFFFFF"


def diagram_taxonomy():
    fig, ax = canvas(10, 4.6)
    box(ax, 24, 38, 52, 6, "Hệ gợi ý dùng đồ thị tri thức", fc=NAVY, color="white", bold=True, fs=11)
    cols = [
        (2, "Dựa trên embedding", "Học vector cho thực thể,\nquan hệ (TransE, TransR)\nrồi đưa vào mô hình gợi ý", "CKE, DKN, KTUP", LIGHT),
        (35.5, "Dựa trên đường dẫn", "Khai thác meta-path nối\nngười dùng với sản phẩm\ntrên đồ thị", "PER, MCRec, KPRN", AMBER),
        (69, "Dựa trên lan truyền", "Lan truyền qua nhiều bước\nláng giềng, học trọng số\nbằng attention", "RippleNet, KGCN,\nKGAT, CKAN", GREEN),
    ]
    for x, title, desc, models, fc in cols:
        box(ax, x, 23, 29, 8, title, fc=fc, bold=True, fs=10)
        ax.text(x + 14.5, 16, desc, ha="center", va="center", fontsize=8.4, color=SLATE, linespacing=1.3)
        box(ax, x + 2, 1.5, 25, 7.5, models, fc=WHITE, ec=GRAY, fs=8.8)
        arrow(ax, (50, 38), (x + 14.5, 31))
    save(fig, "diagram_taxonomy")


def diagram_ripple_sets():
    fig, ax = canvas(10, 4.0)
    box(ax, 2, 14, 17, 12, "Lịch sử nhấp chuột\ncủa người dùng u\n$\\mathcal{V}_u$", fc=AMBER, fs=9.5)
    box(ax, 28, 14, 20, 12, "Ripple set bước 1\n$\\mathcal{S}_u^1=\\{(h,r,t)\\ |\\ h\\in\\mathcal{E}_u^0\\}$", fc=LIGHT, fs=9.2)
    box(ax, 57, 14, 20, 12, "Ripple set bước 2\n$\\mathcal{S}_u^2=\\{(h,r,t)\\ |\\ h\\in\\mathcal{E}_u^1\\}$", fc=LIGHT, fs=9.2)
    box(ax, 85, 14, 13, 12, "...\nbước H", fc=WHITE, ec=GRAY, fs=9.5)
    arrow(ax, (19, 20), (28, 20), "hạt giống\n$\\mathcal{E}_u^0=\\mathcal{V}_u$", dy=1.6)
    arrow(ax, (48, 20), (57, 20), "đuôi $\\mathcal{E}_u^1$", dy=1.6)
    arrow(ax, (77, 20), (85, 20), "đuôi $\\mathcal{E}_u^2$", dy=1.6)
    for x, strength in [(38, "mạnh"), (67, "yếu dần"), (91.5, "yếu hơn nữa")]:
        ax.text(x, 9.5, f"tín hiệu sở thích: {strength}", ha="center", fontsize=8.6, color=SLATE, style="italic")
    ax.text(50, 34, "Sở thích lan ra từng lớp trên đồ thị tri thức như gợn sóng trên mặt nước", ha="center", fontsize=10.5, fontweight="bold", color=NAVY)
    save(fig, "diagram_ripple_sets")


def diagram_ripplenet():
    fig, ax = canvas(10, 5.6)
    box(ax, 33, 48, 32, 6, "Embedding item ứng viên  $\\mathbf{v}\\in\\mathbb{R}^d$", fc=AMBER, bold=True, fs=9.2)
    box(ax, 2, 0.5, 22, 6, "Lịch sử $\\mathcal{V}_u$ (hạt giống)", fc=AMBER, fs=9)
    for k, y in [(1, 30), (2, 11)]:
        box(ax, 2, y, 22, 11, f"Ripple set bước {k}\n$(h_i, r_i, t_i)\\in\\mathcal{{S}}_u^{k}$", fc=LIGHT, ls=1.5)
        box(ax, 38, y + 5.8, 22, 5.2, "$p_i=\\mathrm{softmax}(\\mathbf{v}^{\\top}\\mathbf{R}_i\\mathbf{h}_i)$", fc=WHITE, fs=9)
        box(ax, 38, y, 22, 5.2, "$\\mathbf{o}_u^{%d}=\\sum_i p_i\\,\\mathbf{t}_i$" % k, fc=GREEN, fs=9.5)
        arrow(ax, (24, y + 8.4), (38, y + 8.4))
        ax.text(34.2, y + 9.0, "$\\mathbf{R}_i,\\mathbf{h}_i$", ha="center", va="bottom", fontsize=8.5, color=SLATE)
        arrow(ax, (24, y + 2.6), (38, y + 2.6))
        ax.text(34.2, y + 3.2, "$\\mathbf{t}_i$", ha="center", va="bottom", fontsize=8.5, color=SLATE)
        ax.plot([60, 66], [y + 2.6, y + 2.6], color=SLATE, lw=1.2)
        arrow(ax, (66, y + 2.6), (72, 21.5 if k == 2 else 25.5))
    arrow(ax, (13, 6.5), (13, 11))
    arrow(ax, (13, 30), (13, 22))
    ax.text(14.2, 26, "đuôi bước 1\nlà đầu bước 2", fontsize=7.4, ha="left", va="center", color=SLATE, style="italic")
    arrow(ax, (49, 48), (49, 41))
    elbow(ax, [(33, 51), (30, 51), (30, 19.4), (38, 19.4)])
    ax.text(30, 36, "$\\mathbf{v}$", fontsize=9.5, ha="right", color=SLATE)
    box(ax, 72, 19, 26, 9, "$\\mathbf{u}=\\mathbf{o}_u^1+\\mathbf{o}_u^2+\\dots+\\mathbf{o}_u^H$", fc=GREEN, fs=9.6)
    box(ax, 72, 2, 26, 8.5, "$\\hat{y}_{uv}=\\sigma(\\mathbf{u}^{\\top}\\mathbf{v})$", fc=NAVY, color="white", bold=True, fs=11)
    arrow(ax, (85, 19), (85, 10.5))
    elbow(ax, [(65, 51), (99.3, 51), (99.3, 6.2), (98.3, 6.2)])
    ax.text(80, 52, "$\\mathbf{v}$", fontsize=9.5, ha="center", color=SLATE)
    ax.text(49, 7.2, "ở bước $k>1$, bài báo thay $\\mathbf{v}$ bằng $\\mathbf{o}_u^{k-1}$ khi tính $p_i$", ha="center", fontsize=7.8, color=SLATE, style="italic")
    save(fig, "diagram_ripplenet")


def diagram_ckan():
    fig, ax = canvas(10, 8.0)
    ax.text(24, 77.5, "NHÁNH NGƯỜI DÙNG", ha="center", fontsize=11, fontweight="bold", color=NAVY)
    ax.text(76, 77.5, "NHÁNH ITEM", ha="center", fontsize=11, fontweight="bold", color="#047857")
    for x, fc, ec, seed, init, tag in [
        (4, LIGHT, NAVY, "Lịch sử tương tác của u", "$\\mathcal{E}_u^0$: các item u đã tương tác", "u"),
        (56, GREEN, "#047857", "Item ứng viên v", "$\\mathcal{E}_v^0$: v và các item cùng người dùng", "v"),
    ]:
        box(ax, x, 68, 40, 6, seed, fc=AMBER, ec="#B45309", bold=True)
        box(ax, x, 55.5, 40, 9, "Lan truyền cộng tác\n" + init, fc=fc, ec=ec, fs=9, ls=1.6)
        box(ax, x, 43, 40, 9, "Lan truyền tri thức,  $l=1,\\dots,L$\n$\\mathcal{S}_%s^l=\\{(h,r,t)\\ |\\ h\\in\\mathcal{E}_%s^{l-1}\\}$" % (tag, tag), fc=fc, ec=ec, fs=9, ls=1.6)
        box(ax, x, 29, 40, 10.5, "Lớp attention nhận biết tri thức\n$\\tilde{\\pi}_i=\\mathrm{softmax}\\,\\mathrm{MLP}([\\mathbf{e}_{h_i}\\Vert\\mathbf{r}_i])$,   $\\mathbf{e}_%s^{(l)}=\\sum_i\\tilde{\\pi}_i\\mathbf{e}_{t_i}$" % tag, fc=WHITE, ec=ec, fs=9, ls=1.9)
        box(ax, x, 16.5, 40, 9, "Bộ tổng hợp (concat)\n$\\mathbf{e}_%s=[\\mathbf{e}_%s^{(0)}\\Vert\\mathbf{e}_%s^{(1)}\\Vert\\dots\\Vert\\mathbf{e}_%s^{(L)}]$" % (tag, tag, tag, tag), fc=fc, ec=ec, fs=9, ls=1.7)
        for a, b in [(68, 64.5), (55.5, 52), (43, 39.5), (29, 25.5)]:
            arrow(ax, (x + 20, a), (x + 20, b))
    box(ax, 30, 3, 40, 8, "$\\hat{y}_{uv}=\\sigma(\\mathbf{e}_u^{\\top}\\mathbf{e}_v)$", fc=NAVY, color="white", bold=True, fs=12)
    arrow(ax, (24, 16.5), (42, 11))
    arrow(ax, (76, 16.5), (58, 11))
    ax.text(50, 34, "dùng chung\nbảng embedding\nvà trọng số MLP", ha="center", va="center", fontsize=8.4, color=SLATE, style="italic")
    save(fig, "diagram_ckan")


def diagram_attention():
    fig, ax = canvas(10, 3.6)
    box(ax, 1, 20, 13, 9, "$[\\mathbf{e}_h\\Vert\\mathbf{r}]$\n$\\in\\mathbb{R}^{2d}$", fc=AMBER)
    steps = [("Linear $2d\\to d$\n+ ReLU", LIGHT), ("Linear $d\\to d$\n+ ReLU", LIGHT), ("Linear $d\\to 1$\n+ Sigmoid", LIGHT), ("Softmax trên\ncả tập bộ ba", GREEN)]
    x = 19
    for text, fc in steps:
        box(ax, x, 20, 14, 9, text, fc=fc, fs=9.2)
        arrow(ax, (x - 5, 24.5), (x, 24.5))
        x += 19
    box(ax, 80, 4, 18, 9, "$\\mathbf{e}^{(l)}=\\sum_i\\tilde{\\pi}_i\\,\\mathbf{e}_{t_i}$", fc=NAVY, color="white", bold=True, fs=10.5)
    arrow(ax, (83, 20), (89, 13), "$\\tilde{\\pi}_i$", dy=0.4, fs=9)
    box(ax, 55, 4, 16, 9, "Embedding đuôi $\\mathbf{e}_{t_i}$", fc=AMBER, fs=9.2)
    arrow(ax, (71, 8.5), (80, 8.5))
    ax.text(47, 33, "$\\pi(\\mathbf{e}_h,\\mathbf{r})$: mức độ quan trọng của bộ ba, phụ thuộc cả thực thể đầu lẫn quan hệ", ha="center", fontsize=9.6, color=SLATE, style="italic")
    save(fig, "diagram_attention")


def diagram_collab():
    fig, ax = canvas(10, 4.6)
    ax.text(25, 43, "Tập thực thể khởi đầu của người dùng u", ha="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(75, 43, "Tập thực thể khởi đầu của item v", ha="center", fontsize=10.5, fontweight="bold", color="#047857")
    box(ax, 3, 22, 10, 7, "u", fc=AMBER, bold=True, fs=12)
    for i, y in enumerate([33, 22, 11]):
        box(ax, 28, y, 16, 6.5, f"item $v_{i + 1}$ đã thích", fc=LIGHT, fs=9.2)
        arrow(ax, (13, 25.5), (28, y + 3.2))
    ax.text(36, 5, "$\\mathcal{E}_u^0=\\{e\\ |\\ e=\\varphi(v),\\ y_{uv}=1\\}$", ha="center", fontsize=10, color=NAVY)
    box(ax, 53, 22, 10, 7, "v", fc=AMBER, bold=True, fs=12)
    for i, y in enumerate([31, 13]):
        box(ax, 68, y, 10, 6.5, f"$u_{i + 1}$", fc=WHITE, ec=GRAY, fs=10)
        arrow(ax, (63, 25.5), (68, y + 3.2))
    for i, (y, src) in enumerate([(36, 34.2), (26, 34.2), (16, 16.2), (6, 16.2)]):
        box(ax, 84, y, 14, 6, f"item $v_u^{{{i + 1}}}$", fc=GREEN, ec="#047857", fs=9.2)
        arrow(ax, (78, src), (84, y + 3))
    ax.text(59.5, 35, "những người\nđã thích v", ha="center", fontsize=8, color=SLATE, style="italic")
    ax.text(75, 1.2, "$\\mathcal{E}_v^0=\\{e\\ |\\ e=\\varphi(v_u)\\}\\cup\\{\\varphi(v)\\}$", ha="center", fontsize=10, color="#047857")
    save(fig, "diagram_collab")


def diagram_data_pipeline():
    fig, ax = canvas(10, 3.4)
    steps = [
        ("Dữ liệu tương tác\ngốc của ba tập", AMBER),
        ("Ánh xạ item\nsang thực thể KG", LIGHT),
        ("Nhị phân hoá nhãn\n(phim: rating ≥ 4)", LIGHT),
        ("Lấy mẫu âm 1:1\ntừ item chưa\ntương tác", LIGHT),
        ("ratings_final.npy\nkg_final.npy", GREEN),
    ]
    x = 1
    for i, (text, fc) in enumerate(steps):
        box(ax, x, 12, 17, 13, text, fc=fc, fs=8.2, ls=1.4)
        if i:
            arrow(ax, (x - 3.4, 18.5), (x, 18.5))
        x += 20.4
    ax.text(50, 30, "Tiền xử lý theo quy trình của mã nguồn gốc RippleNet và CKAN", ha="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(50, 5, "Riêng tập phim: lấy mẫu ngẫu nhiên 2.500 người dùng để vừa bộ nhớ khi huấn luyện", ha="center", fontsize=9, color=SLATE, style="italic")
    save(fig, "diagram_data_pipeline")


def diagram_experiment():
    fig, ax = canvas(10, 4.6)
    box(ax, 2, 19, 15, 9, "ratings_final\n+ kg_final", fc=AMBER, bold=True)
    box(ax, 23, 19, 15, 9, "Chia 6 : 2 : 2\ntrain / val / test", fc=LIGHT)
    arrow(ax, (17, 23.5), (23, 23.5))
    models = [("MostPopular", 37), ("Matrix Factorization", 27.5), ("RippleNet", 18), ("CKAN", 8.5)]
    for name, y in models:
        box(ax, 45, y, 19, 6.5, name, fc=WHITE if name != "CKAN" else GREEN, bold=name in ("RippleNet", "CKAN"))
        arrow(ax, (38, 23.5), (45, y + 3.2))
    box(ax, 45, 0.5, 19, 5, "Dựng tập bộ ba của\nngười dùng và item", fc=WHITE, ec=GRAY, fs=7.8)
    arrow(ax, (54.5, 5.5), (54.5, 8.5), color=GRAY)
    box(ax, 71, 30, 27, 9, "Dự đoán CTR\nROC-AUC, F1, Accuracy", fc=LIGHT)
    box(ax, 71, 18.5, 27, 9, "Xếp hạng Top-K\nRecall@{5,10,20,50,100}", fc=LIGHT)
    box(ax, 71, 7, 27, 9, "Độ thưa: giữ 10% đến 100%\ntập huấn luyện, đo ROC-AUC", fc=LIGHT)
    for y in (34.5, 23, 11.5):
        arrow(ax, (64, 23.5), (71, y))
    ax.text(31, 14, "chọn checkpoint có\nval AUC cao nhất", ha="center", fontsize=8.4, color=SLATE, style="italic")
    save(fig, "diagram_experiment")


def diagram_system():
    fig, ax = canvas(10, 5.4)
    box(ax, 30, 45, 40, 6.5, "Trình duyệt người dùng", fc=AMBER, bold=True)
    box(ax, 22, 32, 56, 8.5, "Frontend: React 18 + Vite + Tailwind CSS\nGợi ý Top-K · Lý giải · Đồ thị tri thức · Thư viện", fc=LIGHT, fs=9.4)
    box(ax, 22, 18, 56, 9.5, "Backend: FastAPI\nBộ máy gợi ý (PyTorch CKAN) · Lý giải · Benchmark", fc=GREEN, ec="#047857", fs=9.4)
    arrow(ax, (50, 45), (50, 40.5))
    arrow(ax, (50, 32), (50, 27.5))
    ax.text(52, 29.7, "REST /api/v1", ha="left", va="center", fontsize=8.6, color=SLATE, style="italic")
    stores = [
        (1, "PostgreSQL\nngười dùng, đánh giá,\ntương tác (tập phim)"),
        (26, "Neo4j\nđồ thị con quanh người\ndùng (tập phim)"),
        (51, "saved_models/\ntrọng số CKAN, cấu hình,\nitem_embeddings.npy"),
        (76, "backend/data/\nkg_final, ratings_final,\nmetadata, tên thực thể"),
    ]
    for x, text in stores:
        box(ax, x, 2, 23, 10.5, text, fc=WHITE, ec=GRAY, fs=8.5)
        arrow(ax, (50, 18), (x + 11.5, 12.5), color=GRAY)
    save(fig, "diagram_system")


def diagram_inference():
    fig, ax = canvas(10, 4.4)
    ax.text(50, 41.5, "Ngoại tuyến (một lần, do notebook xuất ra)", ha="center", fontsize=10, fontweight="bold", color="#047857")
    box(ax, 6, 31, 24, 7, "Tập bộ ba của từng item", fc=GREEN, ec="#047857", fs=9)
    box(ax, 38, 31, 24, 7, "Attention + concat", fc=GREEN, ec="#047857", fs=9)
    box(ax, 70, 31, 26, 7, "Ma trận embedding\nitem $\\mathbf{E}_V$", fc=GREEN, ec="#047857", fs=9)
    arrow(ax, (30, 34.5), (38, 34.5))
    arrow(ax, (62, 34.5), (70, 34.5))
    ax.text(50, 24.5, "Trực tuyến (mỗi yêu cầu gợi ý)", ha="center", fontsize=10, fontweight="bold", color=NAVY)
    steps = [("Các item\nngười dùng\nđã thích", AMBER), ("Dựng tập bộ ba\nngười dùng", LIGHT), ("Attention\n+ concat\n$\\rightarrow\\ \\mathbf{e}_u$", LIGHT), ("$\\sigma(\\mathbf{E}_V\\,\\mathbf{e}_u)$\ntoàn danh mục", LIGHT), ("Top-K + lý do\ntừ đường dẫn", NAVY)]
    x = 1
    for i, (text, fc) in enumerate(steps):
        box(ax, x, 7, 17, 12.5, text, fc=fc, fs=8.7, color="white" if fc == NAVY else "#0F172A", bold=fc == NAVY)
        if i:
            arrow(ax, (x - 3.4, 13.2), (x, 13.2))
        x += 20.4
    arrow(ax, (83, 31), (70, 19.5), color="#047857", rad=-0.1)
    ax.text(50, 2, "Phần item không phụ thuộc người dùng nên được tính trước", ha="center", fontsize=8.8, color=SLATE, style="italic")
    save(fig, "diagram_inference")


def diagram_kg_example():
    """A real explanation subgraph, fetched from the running backend."""
    url = "http://127.0.0.1:8000/api/v1/explainability/234?domain=movie&userId=1"
    data = json.load(urllib.request.urlopen(url, timeout=30))
    paths = [p for p in data["paths"] if "#" not in p["entityName"]][:4]
    if not paths:
        raise SystemExit("No named explanation path returned by the backend")
    sources = list(dict.fromkeys(p["sourceMovieTitle"] for p in paths))
    entities = list(dict.fromkeys((p["entityName"], p["relationLabel"]) for p in paths))
    target = paths[0]["targetMovieTitle"]
    (OUT / "kg_example.json").write_text(json.dumps({"request": url, "paths": paths}, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = canvas(10, 3.4)
    spread = lambda n: [30 * (i + 1) / (n + 1) + 1 for i in range(n)][::-1]
    mid = 16.5
    box(ax, 0.5, mid - 3.5, 13, 7, f"Người dùng #{data['userId']}", fc=AMBER, bold=True, fs=9.3)
    src_y = dict(zip(sources, spread(len(sources))))
    ent_y = dict(zip([e for e, _ in entities], spread(len(entities))))
    for title, y in src_y.items():
        box(ax, 21, y - 3, 19, 6, title, fc=ROSE, ec="#B91C1C", fs=9)
        arrow(ax, (13.5, mid), (21, y))
    for (name, rel), y in zip(entities, ent_y.values()):
        box(ax, 54, y - 3, 17, 6, name, fc=WHITE, ec=GRAY, fs=9)
        arrow(ax, (85, mid), (71, y))
        ax.text(62.5, y + 5, f"quan hệ: {rel}", ha="center", fontsize=8.2, color=SLATE, style="italic")
    box(ax, 85, mid - 3.5, 14.5, 7, target, fc=LIGHT, bold=True, fs=9.3)
    for path in paths:
        arrow(ax, (40, src_y[path["sourceMovieTitle"]]), (54, ent_y[path["entityName"]]))
    ax.text(7, 1, "người dùng", ha="center", fontsize=8.4, color="#B45309")
    ax.text(30.5, 1, "phim đã thích", ha="center", fontsize=8.4, color="#B91C1C")
    ax.text(62.5, 1, "thực thể chung trong KG", ha="center", fontsize=8.4, color=SLATE)
    ax.text(92, 1, "phim được gợi ý", ha="center", fontsize=8.4, color=NAVY)
    save(fig, "diagram_kg_example")


# --------------------------------------------------------------------------- equations
EQUATIONS = {
    "mf": r"\hat{y}_{uv}=\sigma\left(\mathbf{u}_u^{\top}\mathbf{v}_v+b_u+b_v\right)",
    "kg": r"\mathcal{G}=\{(h,r,t)\ |\ h,t\in\mathcal{E},\ r\in\mathcal{R}\}",
    "task": r"\hat{y}_{uv}=\mathcal{F}(u,v\ |\ \Theta,\mathbf{Y},\mathcal{G})",
    "rn_entity": r"\mathcal{E}_u^{k}=\{t\ |\ (h,r,t)\in\mathcal{G},\ h\in\mathcal{E}_u^{k-1}\},\quad k=1,2,\dots,H,\qquad \mathcal{E}_u^{0}=\mathcal{V}_u=\{v\ |\ y_{uv}=1\}",
    "rn_set": r"\mathcal{S}_u^{k}=\{(h,r,t)\ |\ (h,r,t)\in\mathcal{G},\ h\in\mathcal{E}_u^{k-1}\},\quad k=1,2,\dots,H",
    "rn_p": r"p_i=\mathrm{softmax}\left(\mathbf{v}^{\top}\mathbf{R}_i\mathbf{h}_i\right)=\frac{\exp\left(\mathbf{v}^{\top}\mathbf{R}_i\mathbf{h}_i\right)}{\sum_{(h,r,t)\in\mathcal{S}_u^{1}}\exp\left(\mathbf{v}^{\top}\mathbf{R}\mathbf{h}\right)}",
    "rn_o": r"\mathbf{o}_u^{1}=\sum_{(h_i,r_i,t_i)\in\mathcal{S}_u^{1}}p_i\,\mathbf{t}_i",
    "rn_u": r"\mathbf{u}=\mathbf{o}_u^{1}+\mathbf{o}_u^{2}+\dots+\mathbf{o}_u^{H}",
    "rn_y": r"\hat{y}_{uv}=\sigma\left(\mathbf{u}^{\top}\mathbf{v}\right)",
    "rn_loss": r"\mathcal{L}=\sum_{(u,v)\in\mathbf{Y}}-\left(y_{uv}\log\hat{y}_{uv}+(1-y_{uv})\log(1-\hat{y}_{uv})\right)+\frac{\lambda_2}{2}\sum_{r\in\mathcal{R}}\Vert\mathbf{I}_r-\mathbf{E}^{\top}\mathbf{R}\mathbf{E}\Vert_2^2+\frac{\lambda_1}{2}\left(\Vert\mathbf{V}\Vert_2^2+\Vert\mathbf{E}\Vert_2^2+\sum_{r}\Vert\mathbf{R}\Vert_2^2\right)",
    "rn_impl_p": r"\mathbf{v}^{(k)}=\mathbf{M}^{\top}\left(\mathbf{v}^{(k-1)}+\mathbf{o}_u^{k}\right),\quad \mathbf{v}^{(0)}=\mathbf{v},\qquad \hat{y}_{uv}=\sigma\left(\left(\mathbf{o}_u^{1}+\dots+\mathbf{o}_u^{H}\right)^{\top}\mathbf{v}^{(H)}\right)",
    "rn_impl_loss": r"\mathcal{L}=\mathrm{BCE}\left(y,\hat{y}\right)-\lambda_{KGE}\sum_{k=1}^{H}\mathrm{mean}_{(h,r,t)\in\mathcal{S}_u^{k}}\,\sigma\left(\mathbf{h}^{\top}\mathbf{R}\mathbf{t}\right)+\lambda_{L2}\sum_{k=1}^{H}\left(\Vert\mathbf{h}\Vert_2^2+\Vert\mathbf{t}\Vert_2^2+\Vert\mathbf{R}\Vert_2^2\right)",
    "ck_eu0": r"\mathcal{E}_u^{0}=\{e\ |\ e=\varphi(v),\ v\in\{v\ |\ y_{uv}=1\}\}",
    "ck_ev0": r"\mathcal{E}_v^{0}=\{e\ |\ e=\varphi(v_u),\ v_u\in\mathcal{V}_v\}\cup\{\varphi(v)\},\qquad \mathcal{V}_v=\{v_u\ |\ \exists u:\ y_{uv}=1\ \wedge\ y_{uv_u}=1\}",
    "ck_prop": r"\mathcal{E}_o^{l}=\{t\ |\ (h,r,t)\in\mathcal{G},\ h\in\mathcal{E}_o^{l-1}\},\qquad \mathcal{S}_o^{l}=\{(h,r,t)\ |\ (h,r,t)\in\mathcal{G},\ h\in\mathcal{E}_o^{l-1}\},\qquad l=1,\dots,L",
    "ck_a": r"\mathbf{a}_i=\pi\left(\mathbf{e}_i^{h},\mathbf{r}_i\right)\mathbf{e}_i^{t}",
    "ck_mlp": r"\mathbf{z}_0=\mathrm{ReLU}\left(\mathbf{W}_0\left(\mathbf{e}_i^{h}\Vert\mathbf{r}_i\right)+\mathbf{b}_0\right),\qquad \pi\left(\mathbf{e}_i^{h},\mathbf{r}_i\right)=\sigma\left(\mathbf{W}_2\,\mathrm{ReLU}\left(\mathbf{W}_1\mathbf{z}_0+\mathbf{b}_1\right)+\mathbf{b}_2\right)",
    "ck_soft": r"\tilde{\pi}\left(\mathbf{e}_i^{h},\mathbf{r}_i\right)=\frac{\exp\left(\pi\left(\mathbf{e}_i^{h},\mathbf{r}_i\right)\right)}{\sum_{(h',r',t')\in\mathcal{S}_o^{l}}\exp\left(\pi\left(\mathbf{e}^{h'},\mathbf{r}'\right)\right)}",
    "ck_layer": r"\mathbf{e}_o^{(l)}=\sum_{i=1}^{|\mathcal{S}_o^{l}|}\tilde{\pi}\left(\mathbf{e}_i^{h},\mathbf{r}_i\right)\mathbf{e}_i^{t},\qquad l=1,\dots,L",
    "ck_zero": r"\mathbf{e}_o^{(0)}=\frac{1}{|\mathcal{E}_o^{0}|}\sum_{e\in\mathcal{E}_o^{0}}\mathbf{e}",
    "ck_sets": r"\mathcal{T}_u=\{\mathbf{e}_u^{(0)},\mathbf{e}_u^{(1)},\dots,\mathbf{e}_u^{(L)}\},\qquad \mathcal{T}_v=\{\mathbf{e}_v^{(origin)},\mathbf{e}_v^{(0)},\mathbf{e}_v^{(1)},\dots,\mathbf{e}_v^{(L)}\}",
    "ck_agg_sum": r"\mathrm{agg}_{sum}=\phi\left(\sum_{\mathbf{e}\in\mathcal{T}_o}\mathbf{W}_a\mathbf{e}+\mathbf{b}_a\right),\qquad \mathrm{agg}_{pool}=\phi\left(\mathrm{pool}_{max}\{\mathbf{W}_a\mathbf{e}+\mathbf{b}_a\}\right)",
    "ck_agg_cat": r"\mathrm{agg}_{concat}=\phi\left(\mathbf{W}_a\left(\mathbf{e}_o^{(i_1)}\Vert\mathbf{e}_o^{(i_2)}\Vert\dots\Vert\mathbf{e}_o^{(i_n)}\right)+\mathbf{b}_a\right)",
    "ck_y": r"\hat{y}_{uv}=\sigma\left(\mathbf{e}_u^{\top}\mathbf{e}_v\right)",
    "ck_loss": r"\mathcal{L}=\sum_{u\in\mathcal{U}}\left(\sum_{v:\,y_{uv}=1}\mathcal{J}(y_{uv},\hat{y}_{uv})-\sum_{i=1}^{|\mathcal{N}_u|}\mathbb{E}_{v_i\sim P(v_i)}\mathcal{J}(y_{uv_i},\hat{y}_{uv_i})\right)+\lambda\Vert\Theta\Vert_2^2",
    "ck_impl": r"\mathbf{e}_u=\left[\mathbf{e}_u^{(0)}\Vert\mathbf{e}_u^{(1)}\Vert\dots\Vert\mathbf{e}_u^{(L)}\right],\qquad \mathbf{e}_v=\left[\mathbf{e}_v\Vert\mathbf{e}_v^{(1)}\Vert\dots\Vert\mathbf{e}_v^{(L)}\right],\qquad \hat{y}_{uv}=\sigma\left(\mathbf{e}_u^{\top}\mathbf{e}_v\right)",
    "auc": r"\mathrm{AUC}=\frac{1}{|\mathcal{P}||\mathcal{N}|}\sum_{p\in\mathcal{P}}\sum_{n\in\mathcal{N}}\mathbb{1}\left[\hat{y}_p>\hat{y}_n\right]",
    "f1": r"F_1=\frac{2\cdot\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}},\qquad \mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}",
    "recall": r"\mathrm{Recall@}K=\frac{|\mathrm{Top}_K(u)\cap\mathcal{T}_u^{test}|}{|\mathcal{T}_u^{test}|}",
    "pop": r"s(v)=\frac{\mathrm{Count}(v)}{\max_{v'}\mathrm{Count}(v')}",
}


def render_equations():
    for key, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, f"${tex}$", fontsize=13)
        fig.savefig(EQ / f"{key}.png", dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=False)
        plt.close(fig)
    print("equations", len(EQUATIONS))


if __name__ == "__main__":
    render_equations()
    chart_ctr()
    chart_recall()
    chart_sparsity()
    chart_gain()
    chart_relations()
    for diagram in (diagram_taxonomy, diagram_ripple_sets, diagram_ripplenet, diagram_ckan, diagram_attention,
                    diagram_collab, diagram_data_pipeline, diagram_experiment, diagram_system, diagram_inference,
                    diagram_kg_example):
        diagram()
    chart_kg_structure()
