#!/usr/bin/env python3
"""
Cross-dataset evaluation at IR = 10.
Run after ../fcnn_ir10_tuned and ../clean_keywords_experiment (needs their .pt models and test CSVs).

Models (both: 141 -> 4x600 -> 1, dropout 0.5, early stopping on val loss patience 10, seed 42, trained on 2019):
  A  "clean-kw model"   trained on features built with the peer's 3,973 clean keywords
  B  "top3000 model"    trained on features built with our top-3000 keywords
Test sets (both: base year 2022, features 2020-22, label = >=10 citations from new papers 2023-25):
  old  top-3000 test set, 86,316 pairs
  new  clean-keyword test set, 36,431 pairs
The 141 features are vocabulary-agnostic (degrees, citations, ranks, similarity indices), so any model
can score any test set. Each model uses ITS OWN log1p + standardisation scaler (stored in its checkpoint).
Main question: does the clean-kw model still work on the larger old test set (A on old vs B on old)?
Paired bootstrap (2000 resamples) for differences on the same test set.
"""
import json, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")                       # repo root
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
CK = os.path.join(ROOT, "clean_keywords_experiment")
FP = os.path.join(ROOT, "feature_pipeline", "output")
MODELS = {"clean-kw model": os.path.join(CK, "results", "fcnn_ir10_cleankw.pt"),
          "top3000 model": os.path.join(ROOT, "fcnn_ir10_tuned", "results", "fcnn_ir10_tuned.pt")}
TESTS = {"old test (top-3000, 86k)": os.path.join(ROOT, "fcnn_ir10", "data", "balanced_ir10_test.csv"),
         "new test (clean-kw, 36k)": os.path.join(CK, "data", "balanced_ir10_test.csv")}
LABEL = "ir_ge_10"
INC = {"old test (top-3000, 86k)": os.path.join(FP, "_work", "paper_keyword_incidence.npz"),
       "new test (clean-kw, 36k)": os.path.join(CK, "features", "_work", "paper_keyword_incidence.npz")}
PAPERS = os.path.join(FP, "_work", "papers_citations.parquet")


class FcNN(nn.Module):
    def __init__(self, d_in, hidden=(600, 600, 600, 600), p=0.5):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]; d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load(path):
    ck = torch.load(path, weights_only=False)
    m = FcNN(len(ck["features"]), tuple(ck["hidden"]), ck["dropout"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def score(m, ck, df):
    X = np.log1p(df[ck["features"]].to_numpy(np.float64).clip(min=0))
    X = ((X - ck["mean"]) / ck["std"]).astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(X))).numpy()


models = {k: load(p) for k, p in MODELS.items()}
tests = {}
for k, p in TESTS.items():
    d = pd.read_csv(p); d.columns = d.columns.str.strip(); tests[k] = d

# "both keywords have >= 1 paper by 2022" mask per test set
import scipy.sparse as sp
yr = pd.read_parquet(PAPERS, columns=["year"])["year"].fillna(0).astype(int).to_numpy()
seen = {}
for k, d in tests.items():
    c = np.asarray(sp.load_npz(INC[k]).tocsc()[yr <= 2022].sum(0)).ravel()
    seen[k] = (c[d.u.to_numpy()] > 0) & (c[d.v.to_numpy()] > 0)

rng = np.random.default_rng(0)
S, rows = {}, []
for tk, d in tests.items():
    y = d[LABEL].to_numpy()
    for mk, (m, ck) in models.items():
        s = score(m, ck, d); S[(mk, tk)] = s; yh = (s >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
        b = seen[tk]
        rows.append({"model": mk, "test_set": tk, "rows": len(d), "AUC": round(roc_auc_score(y, s), 4),
                     "AUC_both_keywords_seen": round(roc_auc_score(y[b], s[b]), 4),
                     "accuracy": round(accuracy_score(y, yh), 4), "precision": round(precision_score(y, yh), 4),
                     "recall": round(recall_score(y, yh), 4), "f1": round(f1_score(y, yh), 4),
                     "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                     "top100_hit_rate": round(float(y[np.argsort(-s)[:100]].mean()), 2),
                     "top1000_hit_rate": round(float(y[np.argsort(-s)[:1000]].mean()), 3)})
res = pd.DataFrame(rows)
print(res.to_string(index=False))


def paired_bootstrap(y, a, b, n=2000):
    i_all = np.arange(len(y)); diffs = []
    for _ in range(n):
        i = rng.choice(i_all, len(y), replace=True)
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    diffs = np.array(diffs)
    return {"delta_auc": round(float(roc_auc_score(y, a) - roc_auc_score(y, b)), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]}


boot = {}
for tk, d in tests.items():
    y = d[LABEL].to_numpy()
    boot[f"{tk}: clean-kw model minus top3000 model"] = paired_bootstrap(y, S[("clean-kw model", tk)], S[("top3000 model", tk)])
    print(list(boot)[-1], boot[list(boot)[-1]])
res.to_csv(os.path.join(OUT, "cross_dataset_results.csv"), index=False)
json.dump({"results": rows, "paired_bootstrap": boot}, open(os.path.join(OUT, "cross_dataset_results.json"), "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
for a, tk in zip(ax, tests):
    y = tests[tk][LABEL].to_numpy()
    for mk, c in (("clean-kw model", "C3"), ("top3000 model", "C0")):
        f, t, _ = roc_curve(y, S[(mk, tk)])
        a.plot(f, t, color=c, lw=2, label=f"{mk}  AUC = {roc_auc_score(y, S[(mk, tk)]):.3f}")
    a.plot([0, 1], [0, 1], "k--", lw=1, label="random 0.5")
    a.set(title=f"{tk}\nIR >= 10, features <= 2022 -> impact 2023-25", xlabel="FPR", ylabel="TPR")
    a.legend(loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "cross_dataset_roc.png"), dpi=130)
print("saved to", OUT)
