#!/usr/bin/env python3
"""Cross-dataset check for IR 50 / 100 (seed-42 models of train_multi_ir_cleankw.py): every model scored on
both the clean-keyword and the top-3000 test set of the same IR; paired bootstrap (2000) of the difference."""
import json, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "results")
TEST = {"clean": os.path.join(HERE, "data"), "top3000": os.path.join(ROOT, "fcnn_multi_ir", "data")}

class FcNN(nn.Module):
    def __init__(self, d_in, hidden=(600, 600, 600, 600), p=0.5):
        super().__init__(); layers, d = [], d_in
        for h in hidden: layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]; d = h
        layers.append(nn.Linear(d, 1)); self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

def score(ck, m, df):
    X = ((np.log1p(df[ck["features"]].to_numpy(np.float64).clip(min=0)) - ck["mean"]) / ck["std"]).astype(np.float32)
    with torch.no_grad(): return torch.sigmoid(m(torch.from_numpy(X))).numpy()

rng = np.random.default_rng(0); rows, boot = [], {}
for ir in (50, 100):
    models = {}
    for ds in ("clean", "top3000"):
        ck = torch.load(os.path.join(OUT, f"{ds}_ir{ir}", f"fcnn_{ds}_ir{ir}.pt"), weights_only=False)
        m = FcNN(len(ck["features"])); m.load_state_dict(ck["state_dict"]); m.eval(); models[ds] = (ck, m)
    for tds, path in TEST.items():
        te = pd.read_csv(os.path.join(path, f"balanced_ir{ir}_test.csv")); te.columns = te.columns.str.strip()
        y = te[f"ir_ge_{ir}"].to_numpy(); s = {k: score(ck, m, te) for k, (ck, m) in models.items()}
        for k in s: rows.append({"IR": ir, "model": f"{k} model", "test_set": f"{tds} test", "rows": len(te), "AUC": round(roc_auc_score(y, s[k]), 4)})
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y)); d.append(roc_auc_score(y[i], s["clean"][i]) - roc_auc_score(y[i], s["top3000"][i]))
        boot[f"IR{ir} {tds} test: clean model minus top3000 model"] = {
            "delta": round(roc_auc_score(y, s["clean"]) - roc_auc_score(y, s["top3000"]), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
R = pd.DataFrame(rows); print(R.pivot_table(index=["IR", "model"], columns="test_set", values="AUC").to_string())
for k, v in boot.items(): print(k, v)
R.to_csv(os.path.join(OUT, "cross_dataset_ir50_100.csv"), index=False)
json.dump({"results": rows, "paired_bootstrap": boot}, open(os.path.join(OUT, "cross_dataset_ir50_100.json"), "w"), indent=2)
