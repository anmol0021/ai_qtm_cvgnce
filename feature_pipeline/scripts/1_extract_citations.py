"""Step 1: stream the OpenAlex JSONL and keep, per paper: id, year, title+abstract text,
total citations and yearly citations (counts_by_year) -> output/_work/papers_citations.parquet"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "ai_quantum_imputed.jsonl")
OUT = os.path.join(HERE, "..", "output", "_work"); os.makedirs(OUT, exist_ok=True)
YEARS = list(range(2003, 2027))

def abstract_from_inv(inv):
    if not inv: return ""
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))

rows = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        cby = {c["year"]: c["cited_by_count"] for c in (r.get("counts_by_year") or [])}
        row = {"id": r["id"].rsplit("/", 1)[-1], "year": r.get("publication_year"),
               "text": (r.get("title") or "") + ". " + abstract_from_inv(r.get("abstract_inverted_index")),
               "cited_total": r.get("cited_by_count") or 0}
        for y in YEARS: row[f"c{y}"] = cby.get(y, 0)
        rows.append(row)
df = pd.DataFrame(rows).drop_duplicates("id")
df.to_parquet(os.path.join(OUT, "papers_citations.parquet"), index=False)
yc = df[[f"c{y}" for y in YEARS]].sum()
print(len(df), "papers; citations by year:", yc.to_dict())
print("total vs sum(counts_by_year): ", df.cited_total.sum(), yc.sum())
