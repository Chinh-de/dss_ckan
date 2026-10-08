"""Copy the measured benchmark numbers out of the executed notebook into a JSON file.

The API's sparsity endpoint serves these numbers as-is, so the demo always shows what
the notebook actually measured. Re-run after re-executing the notebook:

    python -m app.scripts.extract_benchmark_results
"""
import json
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
NOTEBOOK = BACKEND_DIR.parent / "notebooks" / "dss-knowledgegraph-for-rs (3).ipynb"
OUT_PATH = BACKEND_DIR / "data" / "benchmark_results.json"

MODELS = ["MostPopular", "MF", "RippleNet", "CKAN"]


def notebook_stdout(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                chunks.append("".join(out.get("text", "")))
    return "\n".join(chunks)


def table(text: str, title: str) -> list:
    """Rows of the first whitespace-separated table printed after a 'BẢNG n' title."""
    section = text[text.index(title):]
    lines = section.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("Dataset"))
    rows = []
    for line in lines[start + 1:]:
        parts = line.split()
        if len(parts) < 3 or parts[0].lower() not in ("movie", "book", "music"):
            break
        rows.append(parts)
    return rows


def main():
    text = notebook_stdout(NOTEBOOK)
    result = {
        d: {"models": {}, "sparsity": {"ratios": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0], "auc": {}}}
        for d in ("movie", "book", "music")
    }

    for dataset, model, auc, f1, acc in table(text, "BẢNG 1"):
        result[dataset.lower()]["models"][model] = {"auc": float(auc), "f1": float(f1), "acc": float(acc), "recall": {}}

    for dataset, model, *recalls in table(text, "BẢNG 2"):
        for k, value in zip((5, 10, 20, 50, 100), recalls):
            result[dataset.lower()]["models"][model]["recall"][str(k)] = float(value)

    for dataset, model, *aucs in table(text, "BẢNG 3"):
        result[dataset.lower()]["sparsity"]["auc"][model] = [float(a) for a in aucs]

    for dataset, block in result.items():
        missing = [m for m in MODELS if m not in block["models"]]
        if missing or len(block["sparsity"]["auc"]) != 3:
            raise SystemExit(f"Notebook output for {dataset} is incomplete: {missing or 'sparsity table'}")
        split = re.search(
            rf"DATASET: {dataset.upper()} ---\s*\n.*?Ratings: ([\d,]+).*?\n.*?Train=([\d,]+) \| Eval=([\d,]+) \| Test=([\d,]+)",
            text,
        )
        if split:
            ratings, train, val, test = (int(g.replace(",", "")) for g in split.groups())
            block["split"] = {"ratings": ratings, "train": train, "eval": val, "test": test}
        # The sparsity experiment is scored on the users who already have a positive at the 10% level.
        subset = re.search(
            rf"DATASET: {dataset.upper()} ---.*?Sparsity \(([\d,]+) người dùng.*?test dùng ([\d,]+) mẫu",
            text,
            re.S,
        )
        if subset:
            users, rows = (int(g.replace(",", "")) for g in subset.groups())
            block["sparsity"]["evalUsers"] = users
            block["sparsity"]["evalRows"] = rows

    payload = {"source": NOTEBOOK.name, "datasets": result}
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(BACKEND_DIR)} from {NOTEBOOK.name}")


if __name__ == "__main__":
    sys.exit(main())
