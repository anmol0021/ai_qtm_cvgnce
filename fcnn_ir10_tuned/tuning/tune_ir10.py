#!/usr/bin/env python3
"""
IR = 10 tuning of dropout and early-stopping criterion for the FcNN.

Everything else fixed (same as the reported model): 141 -> 4x600 -> 1, ReLU, log1p + standardisation,
Adam lr 1e-4 wd 1e-5, batch 1000, BCE, seed 42, 85/15 train/validation split of the 2019 file.

1) For each dropout p in DROPOUTS: train ONCE for MAX_EPOCHS epochs and log, after every epoch,
   validation loss/AUC (2019 hold-out) and test loss/AUC (whole 2022 file).
   The test columns are logged for transparency only, never used for any decision.
2) Early-stopping rules are applied afterwards to these logs. Because training is deterministic and
   a stopping rule only decides WHEN to stop and WHICH epoch to restore, this gives exactly the result
   of running that rule live (e.g. dropout 0.2 + 'val AUC, patience 25' reproduces the reported model).
3) Selection of the best configuration uses VALIDATION AUC only.
"""
import json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "fcnn_ir10", "data")  # same train/test CSVs as fcnn_ir10
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.5]
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 150))
SEED, LABEL, BATCH = 42, "ir_ge_10", 1000

feat = pd.read_csv(os.path.join(DATA, "feature_dictionary.csv")).feature.tolist()
tr_all = pd.read_csv(os.path.join(DATA, "balanced_ir10_train.csv"))
te = pd.read_csv(os.path.join(DATA, "balanced_ir10_test.csv"))
tr_all.columns = tr_all.columns.str.strip(); te.columns = te.columns.str.strip()
tr, va = train_test_split(tr_all, test_size=0.15, stratify=tr_all[LABEL], random_state=SEED)
prep = lambda d: np.log1p(d[feat].to_numpy(np.float64).clip(min=0))
Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
Xtr, Xva, Xte = [torch.from_numpy(((x - mu) / sd).astype(np.float32)) for x in (Xtr, Xva, Xte)]
ytr, yva, yte = [torch.from_numpy(d[LABEL].to_numpy(np.float32)) for d in (tr, va, te)]
print(f"train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)


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
for p in DROPOUTS:
    path = os.path.join(OUT, f"history_dropout{p}.csv")
    if os.path.exists(path) and len(pd.read_csv(path)) >= MAX_EPOCHS:
        print(f"dropout {p}: already done"); continue
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = FcNN(len(feat), p=p)
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
            print(f"dropout {p} epoch {ep:3d}: val_auc {h['val_auc']:.4f} test_auc {h['test_auc']:.4f} [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(hist).to_csv(path, index=False)

# ------------------------------------------------------------------ apply early-stopping rules post hoc
def apply_rule(h, metric, patience, min_delta=1e-4):
    """Live early stopping: track best `metric` on validation; stop after `patience` epochs
    without improvement > min_delta; restore best epoch. patience=None -> train to MAX_EPOCHS."""
    sign = 1 if metric == "val_auc" else -1
    best, best_ep, bad = -np.inf, None, 0
    for _, r in h.iterrows():
        v = sign * r[metric]
        if v > best + min_delta:
            best, best_ep, bad = v, int(r.epoch), 0
        else:
            bad += 1
        if patience is not None and bad >= patience:
            return best_ep, int(r.epoch)
    return best_ep, int(h.epoch.iloc[-1])

RULES = [("val_auc", 5), ("val_auc", 10), ("val_auc", 25), ("val_auc", 50),
         ("val_loss", 5), ("val_loss", 10), ("val_loss", 25), ("val_loss", 50),
         ("none (last epoch)", None)]
rows = []
for p in DROPOUTS:
    h = pd.read_csv(os.path.join(OUT, f"history_dropout{p}.csv"))
    for metric, pat in RULES:
        if metric.startswith("none"):
            best_ep = stop_ep = int(h.epoch.iloc[-1])
        else:
            best_ep, stop_ep = apply_rule(h, metric, pat)
        r = h[h.epoch == best_ep].iloc[0]
        rows.append({"dropout": p, "monitor": metric, "patience": pat, "stopped_at": stop_ep,
                     "restored_epoch": best_ep, "val_auc": round(r.val_auc, 4), "test_auc": round(r.test_auc, 4)})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "grid_results.csv"), index=False)
sel = res.sort_values(["val_auc", "restored_epoch"], ascending=[False, True]).iloc[0]
base = res[(res.dropout == 0.2) & (res.monitor == "val_auc") & (res.patience == 25)].iloc[0]
summary = {"baseline (reported model)": base.to_dict(), "selected by validation AUC": sel.to_dict()}
json.dump(summary, open(os.path.join(OUT, "selection.json"), "w"), indent=2, default=str)
print(res.to_string(index=False)); print(json.dumps(summary, indent=2, default=str))
