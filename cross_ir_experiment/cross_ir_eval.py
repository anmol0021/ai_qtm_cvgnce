#!/usr/bin/env python3
"""
Cross-IR experiment (Gu & Krenn 2024, Fig. 8): is a model trained on IR = 10 better at the
IR = 50 task than a model trained on IR = 50 itself?

Models: the FcNNs trained by ../fcnn_multi_ir/train_multi_ir.py (identical architecture 141 -> 4x600 -> 1
and training parameters; each was trained on its own balanced 2019 set and carries its own
scaler). No retraining: every model scores every balanced 2022 test set (IR 10, 50, 100).
The score is the model's output probability; AUC only needs a ranking, so a model trained for
"impact >= 10" can be evaluated on the "impact >= 50" labels.
Significance: paired bootstrap (2000 resamples of test pairs) of the AUC difference.
"""
import json, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MI = os.path.join(HERE, "..", "fcnn_multi_ir")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
TEST = {ir: os.path.join(MI, "data", f"balanced_ir{ir}_test.csv") for ir in (10, 50, 100)}
IRS = (10, 50, 100)


class FcNN(nn.Module):
    def __init__(self, d_in, hidden=(600, 600, 600, 600), p=0.2):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]; d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load(ir):
    ck = torch.load(os.path.join(MI, "results", f"ir{ir}", f"fcnn_ir{ir}.pt"), weights_only=False)
    m = FcNN(len(ck["features"])); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def score(model, ck, df):
    X = np.log1p(df[ck["features"]].to_numpy(np.float64).clip(min=0))
    X = ((X - ck["mean"]) / ck["std"]).astype(np.float32)
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


models = {ir: load(ir) for ir in IRS}
tests = {}
for ir, p in TEST.items():
    d = pd.read_csv(p); d.columns = d.columns.str.strip(); tests[ir] = d

rng = np.random.default_rng(0)
scores, rows = {}, []
for t_ir, df in tests.items():
    y = df[f"ir_ge_{t_ir}"].to_numpy()
    for m_ir, (m, ck) in models.items():
        s = score(m, ck, df); scores[(m_ir, t_ir)] = s
        rows.append({"trained_on_IR": m_ir, "tested_on_IR": t_ir, "test_rows": len(df),
                     "AUC": round(roc_auc_score(y, s), 4)})
mat = pd.DataFrame(rows)
print("AUC matrix (rows = model trained on IR, columns = test task IR):")
piv = mat.pivot(index="trained_on_IR", columns="tested_on_IR", values="AUC")
print(piv.to_string())


def paired_bootstrap(y, s_a, s_b, n=2000):
    """AUC(a) - AUC(b) on the same resampled pairs."""
    idx_all = np.arange(len(y)); diffs = []
    for _ in range(n):
        i = rng.choice(idx_all, size=len(y), replace=True)
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
    diffs = np.array(diffs)
    return {"delta_auc": round(float(roc_auc_score(y, s_a) - roc_auc_score(y, s_b)), 4),
            "ci95_low": round(float(np.percentile(diffs, 2.5)), 4),
            "ci95_high": round(float(np.percentile(diffs, 97.5)), 4),
            "p_a_not_better": round(float((diffs <= 0).mean()), 4)}


comp = {}
for t_ir, others in ((50, (10,)), (100, (10, 50))):
    y = tests[t_ir][f"ir_ge_{t_ir}"].to_numpy()
    for o in others:
        key = f"test IR{t_ir}: model IR{o} minus model IR{t_ir}"
        comp[key] = paired_bootstrap(y, scores[(o, t_ir)], scores[(t_ir, t_ir)])
        print(key, comp[key])

mat.to_csv(os.path.join(OUT, "cross_ir_auc_matrix.csv"), index=False)
json.dump({"auc_matrix": {f"train{r}_test{c}": float(piv.loc[r, c]) for r in IRS for c in IRS},
           "paired_bootstrap": comp}, open(os.path.join(OUT, "cross_ir_results.json"), "w"), indent=2)

# ---------------------------------------------------------------- figures
col = {10: "C3", 50: "C0", 100: "C2"}
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
y50 = tests[50]["ir_ge_50"].to_numpy()
for m_ir in (50, 10):
    fpr, tpr, _ = roc_curve(y50, scores[(m_ir, 50)])
    ax[0].plot(fpr, tpr, color=col[m_ir], lw=2, label=f"FcNN trained on IR={m_ir}  AUC={roc_auc_score(y50, scores[(m_ir, 50)]):.4f}")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="random AUC=0.5")
ax[0].set(title="Paper Fig. 8 replication: test task IR ≥ 50\n(features ≤2022 → impact 2023–2025)", xlabel="FPR", ylabel="TPR")
ax[0].legend(loc="lower right")
y100 = tests[100]["ir_ge_100"].to_numpy()
for m_ir in (100, 50, 10):
    fpr, tpr, _ = roc_curve(y100, scores[(m_ir, 100)])
    ax[1].plot(fpr, tpr, color=col[m_ir], lw=2, label=f"trained on IR={m_ir}  AUC={roc_auc_score(y100, scores[(m_ir, 100)]):.4f}")
ax[1].plot([0, 1], [0, 1], "k--", lw=1)
ax[1].set(title="Extension: test task IR ≥ 100", xlabel="FPR", ylabel="TPR"); ax[1].legend(loc="lower right")
im = ax[2].imshow(piv.values, cmap="viridis", vmin=piv.values.min() - 0.01, vmax=piv.values.max())
for i in range(3):
    for j in range(3):
        ax[2].text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center",
                   color="white" if piv.values[i, j] < piv.values.mean() else "black", fontsize=12,
                   fontweight="bold" if i == j else "normal")
ax[2].set_xticks(range(3), [f"IR≥{c}" for c in piv.columns]); ax[2].set_yticks(range(3), [f"IR={r}" for r in piv.index])
ax[2].set(title="Test AUC: model (rows) × test task (columns)", xlabel="test task", ylabel="model trained on")
plt.colorbar(im, ax=ax[2], fraction=0.046)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "cross_ir_results.png"), dpi=130)
print("saved to", OUT)
