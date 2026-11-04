#!/usr/bin/env python3
"""
Seed robustness of dropout 0.5 at IR = 10.
Same data, split (random_state 42), preprocessing and model as tune_ir10.py; only the TRAINING seed
(weight init, batch order, dropout masks) changes. Each seed trains 150 epochs and logs per-epoch
val/test loss and AUC to results/seeds/history_dropout0.5_seed{s}.csv (seed 42 = existing run).
Then the stopping rules are applied to every seed's log.
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "fcnn_ir10", "data")  # same train/test CSVs as fcnn_ir10
OUT = os.path.join(HERE, "results", "seeds"); os.makedirs(OUT, exist_ok=True)
SEEDS = [1, 2, 3, 4]
P, MAX_EPOCHS, LABEL, BATCH, SPLIT_SEED = 0.5, 150, "ir_ge_10", 1000, 42

feat = pd.read_csv(os.path.join(DATA, "feature_dictionary.csv")).feature.tolist()
tr_all = pd.read_csv(os.path.join(DATA, "balanced_ir10_train.csv"))
te = pd.read_csv(os.path.join(DATA, "balanced_ir10_test.csv"))
tr_all.columns = tr_all.columns.str.strip(); te.columns = te.columns.str.strip()
tr, va = train_test_split(tr_all, test_size=0.15, stratify=tr_all[LABEL], random_state=SPLIT_SEED)
prep = lambda d: np.log1p(d[feat].to_numpy(np.float64).clip(min=0))
Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
Xtr, Xva, Xte = [torch.from_numpy(((x - mu) / sd).astype(np.float32)) for x in (Xtr, Xva, Xte)]
ytr, yva, yte = [torch.from_numpy(d[LABEL].to_numpy(np.float32)) for d in (tr, va, te)]


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


lossf = nn.BCEWithLogitsLoss()
for s in SEEDS:
    path = os.path.join(OUT, f"history_dropout0.5_seed{s}.csv")
    if os.path.exists(path) and len(pd.read_csv(path)) >= MAX_EPOCHS:
        print(f"seed {s}: already done"); continue
    torch.manual_seed(s); np.random.seed(s)
    model = FcNN(len(feat), p=P)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    hist, t0 = [], time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); loss = lossf(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            lv, lt = model(Xva), model(Xte)
        hist.append({"epoch": ep, "train_loss": tot / len(perm),
                     "val_loss": float(lossf(lv, yva)), "val_auc": roc_auc_score(yva, lv),
                     "test_loss": float(lossf(lt, yte)), "test_auc": roc_auc_score(yte, lt)})
        if ep % 25 == 0 or ep == 1:
            h = hist[-1]
            print(f"seed {s} epoch {ep:3d}: val_auc {h['val_auc']:.4f} val_loss {h['val_loss']:.4f} test_auc {h['test_auc']:.4f} [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(hist).to_csv(path, index=False)

# ------------------------------------------------------------------ apply stopping rules to every seed
def restore(h, metric, patience, min_delta=1e-4):
    """Stop after `patience` epochs without improvement > min_delta; restore the best epoch seen."""
    higher = metric == "val_auc"; s = h[metric].to_numpy() * (1 if higher else -1)
    best, bad, stop = -np.inf, 0, len(s) - 1
    for i, v in enumerate(s):
        if v > best + min_delta: best, bad = v, 0
        else: bad += 1
        if bad >= patience: stop = i; break
    b = int(np.argmax(s[:stop + 1]))
    return b + 1, float(h.val_auc[b]), float(h.test_auc[b])

hists = {42: pd.read_csv(os.path.join(HERE, "results", "history_dropout0.5.csv"))}
for s in SEEDS:
    hists[s] = pd.read_csv(os.path.join(OUT, f"history_dropout0.5_seed{s}.csv"))
RULES = [("val_loss", 10), ("val_loss", 25), ("val_auc", 10), ("val_auc", 25)]
rows = []
for s, h in hists.items():
    for m, pat in RULES:
        ep, v, t = restore(h, m, pat)
        rows.append({"seed": s, "rule": f"{m} patience {pat}", "epoch_used": ep, "val_auc": round(v, 4), "test_auc": round(t, 4)})
    rows.append({"seed": s, "rule": "max test AUC (NOT FAIR)", "epoch_used": int(h.test_auc.idxmax()) + 1,
                 "val_auc": round(float(h.val_auc.max()), 4), "test_auc": round(float(h.test_auc.max()), 4)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "results", "seeds_dropout0.5.csv"), index=False)
summ = res.groupby("rule").test_auc.agg(["mean", "std", "min", "max"]).round(4)
print(res.pivot(index="rule", columns="seed", values="test_auc").to_string())
print("\n", summ.to_string())
json.dump(summ.reset_index().to_dict("records"), open(os.path.join(HERE, "results", "seeds_dropout0.5_summary.json"), "w"), indent=2)
