"""Step 1: top-3000 keywords from the 300 anchor papers using stratum-core's
api/scripts/tfidf_keybert.py, called exactly like the stratum Go server does
(JSON request on stdin -> JSON response on stdout). Script copied unmodified."""
import json, subprocess, sys, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
req = {
    "path": os.path.abspath(os.path.join(HERE, "..", "..", "data", "quantum_x_ai_300_anchors.xlsx.csv")),
    "title_column": "Article Title",
    "abstract_column": "Abstract",
    "doi_column": "DOI",
    "top_n": 3000,
    "use_keybert": True,
    "alpha": 0.10,            # stratum default: 10% TF-IDF + 90% KeyBERT
    "ngram_min": 1, "ngram_max": 3,
    "min_df": 1, "max_df": 0.85,
    "keybert_model": "allenai-specter",
}
p = subprocess.run([sys.executable, os.path.join(HERE, "stratum_tfidf_keybert.py")],
                   input=json.dumps(req), capture_output=True, text=True, encoding="utf-8")
if p.returncode != 0:
    sys.exit(p.stderr[-3000:])
res = json.loads(p.stdout.strip().splitlines()[-1])
print("docs_used:", res["docs_used"], "warnings:", res["warnings"], "n_keywords:", len(res["keywords"]))
out = os.path.join(HERE, "..", "..", "output")
df = pd.DataFrame(res["keywords"])
df.insert(0, "rank", range(1, len(df) + 1))
df.to_csv(os.path.join(out, "stratum_raw_keywords.csv"), index=False)
json.dump(req, open(os.path.join(out, "stratum_request.json"), "w"), indent=2)
