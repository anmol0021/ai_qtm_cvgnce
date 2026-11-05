#!/usr/bin/env python3
"""
How much of the test AUC comes from pairs containing a keyword that has NO paper up to the base year
(2022)? Such pairs have (almost) all-zero features and are mostly negatives, i.e. easy to rank.
Reports AUC on all test pairs and on pairs where both keywords have >= 1 paper by 2022,
for the clean-keyword model and (if available) the top-3000-keyword tuned model.
"""
import json, os
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.join(HERE, "..", "feature_pipeline", "output")
PAPERS = os.path.join(FP, "_work", "papers_citations.parquet")
RUNS = {  # name: (incidence matrix of that keyword list, test predictions)
    "clean keywords": (os.path.join(HERE, "features", "_work", "paper_keyword_incidence.npz"),
                       os.path.join(HERE, "results", "test_predictions_ir10_cleankw.csv")),
    "top-3000 keywords": (os.path.join(FP, "_work", "paper_keyword_incidence.npz"),
                          os.path.join(HERE, "..", "fcnn_ir10_tuned", "results", "test_predictions_ir10_tuned.csv")),
}
BASE = 2022
yr = pd.read_parquet(PAPERS, columns=["year"])["year"].fillna(0).astype(int).to_numpy()
out = {}
for name, (inc, pred) in RUNS.items():
    if not (os.path.exists(inc) and os.path.exists(pred)):
        print(f"{name}: skipped (missing {inc if not os.path.exists(inc) else pred})"); continue
    c = np.asarray(sp.load_npz(inc).tocsc()[yr <= BASE].sum(0)).ravel()
    d = pd.read_csv(pred); y, s = d.ir_ge_10.to_numpy(), d.predicted_prob.to_numpy()
    both = (c[d.u] > 0) & (c[d.v] > 0)
    out[name] = {"keywords": int(len(c)), f"keywords_without_paper_by_{BASE}": int((c == 0).sum()),
                 "test_pairs": int(len(d)), "share_pairs_with_unseen_keyword": round(float((~both).mean()), 3),
                 "positive_share_among_those": round(float(y[~both].mean()), 3),
                 "auc_all": round(roc_auc_score(y, s), 4),
                 "auc_both_keywords_seen": round(roc_auc_score(y[both], s[both]), 4)}
    print(name, out[name])
json.dump(out, open(os.path.join(HERE, "results", "seen_keywords_check.json"), "w"), indent=2)
