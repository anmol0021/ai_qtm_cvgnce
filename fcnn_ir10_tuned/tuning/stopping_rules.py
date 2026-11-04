#!/usr/bin/env python3
"""
Wider set of early-stopping criteria applied to the per-epoch logs of tune_ir10.py (no retraining;
training is deterministic, so this equals running each rule live, within the 150-epoch cap).

Families (all decisions use TRAIN / VALIDATION numbers only):
  patience     : stop after `pat` epochs without val improvement > min_delta (val_auc or val_loss)
  smoothed     : same, on a 5-epoch moving average of val_auc
  GL (Prechelt): generalisation loss GL = 100*(val_loss/min val_loss - 1); stop when GL > alpha
  PQ (Prechelt): GL / training progress P5 (P5 = 1000*(mean last 5 train losses / min of them - 1)); stop when > alpha
  UP (Prechelt): stop when val_loss rose at the end of s successive 5-epoch strips
  fixed        : train exactly N epochs
Each rule is reported twice: 'restore best' (restore the best-validation epoch seen until the stop)
and 'no restore' (keep the weights of the stop epoch).
Selection = highest validation AUC. Test AUC is only reported; the max-test row is labelled NOT FAIR.
"""
import os, json, numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.5]


def stop_patience(s, pat, md, higher):
    best, bad = -np.inf, 0
    for i, v in enumerate(s):
        v = v if higher else -v
        if v > best + md: best, bad = v, 0
        else: bad += 1
        if bad >= pat: return i
    return None


def stop_gl(vl, alpha):
    m = np.minimum.accumulate(vl); gl = 100 * (vl / m - 1)
    hit = np.where(gl > alpha)[0]; return int(hit[0]) if len(hit) else None


def stop_pq(vl, tl, alpha, k=5):
    m = np.minimum.accumulate(vl); gl = 100 * (vl / m - 1)
    for i in range(k - 1, len(vl), k):                 # Prechelt: evaluated at the end of each strip
        w = tl[i - k + 1:i + 1]; p = 1000 * (w.mean() / w.min() - 1)
        if p > 0 and gl[i] / p > alpha: return i
    return None


def stop_up(vl, s, k=5):
    ups = 0
    for i in range(2 * k - 1, len(vl), k):
        ups = ups + 1 if vl[i] > vl[i - k] else 0
        if ups >= s: return i
    return None


rows = []
for p in DROPOUTS:
    h = pd.read_csv(os.path.join(OUT, f"history_dropout{p}.csv"))
    va, vl, tl = h.val_auc.to_numpy(), h.val_loss.to_numpy(), h.train_loss.to_numpy()
    sm = pd.Series(va).rolling(5, min_periods=1).mean().to_numpy()
    rules = {}
    for pat in (1, 2, 3, 5, 10, 25, 50):
        for md in (1e-4, 1e-3, 5e-3):
            rules[f"patience val_auc p={pat} min_delta={md}"] = (stop_patience(va, pat, md, True), "auc")
            rules[f"patience val_loss p={pat} min_delta={md}"] = (stop_patience(vl, pat, md, False), "loss")
    for pat in (5, 10, 25):
        rules[f"smoothed(5) val_auc p={pat}"] = (stop_patience(sm, pat, 1e-4, True), "sm")
    for a in (0.5, 1, 2, 5, 10):
        rules[f"GL alpha={a}"] = (stop_gl(vl, a), "loss")
    for a in (0.25, 0.5, 1, 2):
        rules[f"PQ alpha={a}"] = (stop_pq(vl, tl, a), "loss")
    for s in (1, 2, 3, 4):
        rules[f"UP s={s}"] = (stop_up(vl, s), "loss")
    for n in (5, 10, 20, 30, 50, 75, 100, 150):
        rules[f"fixed {n} epochs"] = (n - 1, None)
    for name, (stop, kind) in rules.items():
        capped = stop is None
        stop = len(h) - 1 if capped else stop
        if kind is None:
            best = stop
        else:
            crit = {"auc": va, "sm": sm, "loss": -vl}[kind][:stop + 1]
            best = int(np.argmax(crit))
        for mode, i in (("restore best", best), ("no restore", stop)):
            if kind is None and mode == "restore best": continue
            rows.append({"dropout": p, "rule": name, "weights": mode, "stopped_at": stop + 1,
                         "hit_150_cap": capped, "epoch_used": i + 1,
                         "val_auc": round(va[i], 4), "test_auc": round(h.test_auc[i], 4)})

res = pd.DataFrame(rows).drop_duplicates(["dropout", "epoch_used"])   # same epoch -> same model
res.to_csv(os.path.join(OUT, "stopping_rules_all.csv"), index=False)
fair = res.sort_values(["val_auc", "epoch_used"], ascending=[False, True]).iloc[0]
cheat = res.sort_values("test_auc", ascending=False).iloc[0]
per_dropout = (res.sort_values(["val_auc", "epoch_used"], ascending=[False, True])
               .groupby("dropout").head(1).sort_values("dropout"))
summary = {"baseline": {"dropout": 0.2, "epoch_used": 82, "val_auc": 0.8637, "test_auc": 0.8598},
           "FAIR: best validation AUC over all rules and dropouts": fair.to_dict(),
           "NOT FAIR: best test AUC (chosen by looking at test)": cheat.to_dict(),
           "distinct models compared": int(len(res))}
json.dump(summary, open(os.path.join(OUT, "stopping_rules_summary.json"), "w"), indent=2, default=str)
print("best validation-selected per dropout:\n", per_dropout.to_string(index=False))
print("\ntest AUC spread per dropout over all rules:")
print(res.groupby("dropout").test_auc.describe()[["count", "min", "50%", "max"]].to_string())
print(json.dumps(summary, indent=2, default=str))
