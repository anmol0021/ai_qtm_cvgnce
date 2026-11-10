#!/usr/bin/env python3
"""
IR = 50 and IR = 100 FcNNs on the CLEAN-KEYWORD datasets, with the tuned IR-10 settings:
141 -> 4x600 -> 1, ReLU, dropout 0.5, log1p + standardisation (fitted on train), Adam lr 1e-4 wd 1e-5,
batch 1000, BCE, max 300 epochs, early stopping on validation LOSS (patience 10, min_delta 1e-4),
restore the lowest-validation-loss epoch; 85/15 stratified train/validation split (random_state 42).
Train = base year 2019 (label 2020-22), test = base year 2022 (label 2023-25).

Because these sets are small (IR 100 test: 1,958 rows), every configuration is trained with 5 seeds
(42, 1, 2, 3, 4); seed 42 is the saved/plotted model, the table reports mean +- sd.
Control: the same settings on the TOP-3000 IR 50 / 100 datasets (../fcnn_multi_ir/data),
so clean vs top-3000 is compared with identical training settings.
Also reported for seed 42: 95% bootstrap CI of the test AUC, and (clean keywords) AUC on test pairs where
both keywords have >= 1 paper by 2022.
"""
import json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, scipy.sparse as sp
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")  # repo root
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
FEAT = pd.read_csv(os.path.join(ROOT, "clean_keywords_experiment", "data", "feature_dictionary.csv")).feature.tolist()
DATA = {"clean": os.path.join(HERE, "data"),
        "top3000": os.path.join(ROOT, "fcnn_multi_ir", "data")}
SEEDS = [42, 1, 2, 3, 4]
torch.set_num_threads(8)


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


def train(tr_all, te, label, seed, patience=10, batch=1000, max_epochs=300):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = train_test_split(tr_all, test_size=0.15, stratify=tr_all[label], random_state=42)
    prep = lambda d: np.log1p(d[FEAT].to_numpy(np.float64).clip(min=0))
    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr, Xva, Xte = [torch.from_numpy(((x - mu) / sd).astype(np.float32)) for x in (Xtr, Xva, Xte)]
    ytr, yva = [torch.from_numpy(d[label].to_numpy(np.float32)) for d in (tr, va)]
    model = FcNN(len(FEAT)); opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    best_loss, ref, bad, best_state, best_ep, hist = np.inf, np.inf, 0, None, 0, []
    for ep in range(1, max_epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); loss = lossf(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            lv = model(Xva)
        vl = float(lossf(lv, yva)); va_auc = roc_auc_score(yva, lv)
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "val_loss": vl, "val_auc": va_auc})
        if vl < best_loss:
            best_loss, best_ep, best_va = vl, ep, va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if vl < ref - 1e-4: ref, bad = vl, 0
        else: bad += 1
        if bad >= patience: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte)).numpy()
    return model, p, {"epochs_run": len(hist), "best_epoch": best_ep, "val_loss": best_loss, "val_auc": best_va,
                      "train_rows": len(tr), "val_rows": len(va), "mean": mu, "std": sd}, pd.DataFrame(hist)


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); a = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): a.append(roc_auc_score(y[i], s[i]))
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


# keyword "seen by 2022" counts for the clean vocabulary
yr = pd.read_parquet(os.path.join(ROOT, "feature_pipeline", "output", "_work", "papers_citations.parquet"),
                     columns=["year"])["year"].fillna(0).astype(int).to_numpy()
seen_cnt = np.asarray(sp.load_npz(os.path.join(ROOT, "clean_keywords_experiment", "features", "_work",
                                               "paper_keyword_incidence.npz")).tocsc()[yr <= 2022].sum(0)).ravel()

summary, preds42, t0 = [], {}, time.time()
for ds in ("clean", "top3000"):
    for ir in (50, 100):
        label = f"ir_ge_{ir}"
        tr_all = pd.read_csv(os.path.join(DATA[ds], f"balanced_ir{ir}_train.csv"))
        te = pd.read_csv(os.path.join(DATA[ds], f"balanced_ir{ir}_test.csv"))
        tr_all.columns = tr_all.columns.str.strip(); te.columns = te.columns.str.strip()
        y = te[label].to_numpy(); aucs = []
        for seed in SEEDS:
            model, p, info, hist = train(tr_all, te, label, seed)
            aucs.append(roc_auc_score(y, p))
            print(f"{ds:7s} IR{ir:3d} seed {seed:2d}: test AUC {aucs[-1]:.4f} (epoch {info['best_epoch']}/{info['epochs_run']}) [{time.time()-t0:.0f}s]", flush=True)
            if seed != 42: continue
            yh = (p >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
            m = {"dataset": ds, "impact_range": ir, "train_rows": info["train_rows"], "val_rows": info["val_rows"],
                 "test_rows": len(te), "dropout": 0.5, "early_stopping": "val_loss, patience 10",
                 "epochs_run": info["epochs_run"], "best_epoch": info["best_epoch"],
                 "val_auc": round(info["val_auc"], 4), "test_auc": round(aucs[-1], 4), "test_auc_ci95": boot_ci(y, p),
                 "test_accuracy": round(accuracy_score(y, yh), 4), "test_precision": round(precision_score(y, yh), 4),
                 "test_recall": round(recall_score(y, yh), 4), "test_f1": round(f1_score(y, yh), 4),
                 "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
                 "top100_hit_rate": round(float(y[np.argsort(-p)[:100]].mean()), 2)}
            if ds == "clean":
                both = (seen_cnt[te.u.to_numpy()] > 0) & (seen_cnt[te.v.to_numpy()] > 0)
                m["share_test_pairs_with_unseen_keyword"] = round(float((~both).mean()), 3)
                m["test_auc_both_keywords_seen"] = round(roc_auc_score(y[both], p[both]), 4)
            d = os.path.join(OUT, f"{ds}_ir{ir}"); os.makedirs(d, exist_ok=True)
            json.dump(m, open(os.path.join(d, f"metrics_{ds}_ir{ir}.json"), "w"), indent=2)
            hist.to_csv(os.path.join(d, f"training_history_{ds}_ir{ir}.csv"), index=False)
            o = te[["u", "v", "keyword_u", "keyword_v", "impact_3y_citations", label]].copy(); o["predicted_prob"] = p
            o.sort_values("predicted_prob", ascending=False).to_csv(os.path.join(d, f"test_predictions_{ds}_ir{ir}.csv"), index=False)
            torch.save({"state_dict": model.state_dict(), "features": FEAT, "mean": info["mean"], "std": info["std"],
                        "hidden": (600, 600, 600, 600), "dropout": 0.5}, os.path.join(d, f"fcnn_{ds}_ir{ir}.pt"))
            preds42[(ds, ir)] = (y, p); m42 = m
        summary.append({"dataset": ds, "IR": ir, "train_rows": len(tr_all), "test_rows": len(te),
                        "test_auc_seed42": round(aucs[0], 4), "ci95_seed42": m42["test_auc_ci95"],
                        "test_auc_mean_5seeds": round(float(np.mean(aucs)), 4), "test_auc_sd_5seeds": round(float(np.std(aucs)), 4),
                        "test_auc_both_seen_seed42": m42.get("test_auc_both_keywords_seen")})

S = pd.DataFrame(summary); S.to_csv(os.path.join(OUT, "summary_ir50_100.csv"), index=False)
print(S.to_string(index=False))

# ROC: clean-keyword models for IR 10 / 50 / 100 (IR 10 = ../clean_keywords_experiment model)
p10 = pd.read_csv(os.path.join(ROOT, "clean_keywords_experiment", "results", "test_predictions_ir10_cleankw.csv"))
curves = {10: (p10.ir_ge_10.to_numpy(), p10.predicted_prob.to_numpy()), 50: preds42[("clean", 50)], 100: preds42[("clean", 100)]}
col = {10: "C3", 50: "C0", 100: "C2"}
fig, ax = plt.subplots(figsize=(6.6, 6))
for ir, (y, p) in curves.items():
    f, t, _ = roc_curve(y, p)
    ax.plot(f, t, color=col[ir], lw=2, label=f"IR ≥ {ir}  AUC = {roc_auc_score(y, p):.3f}  (n = {len(y):,})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="random 0.5")
ax.set(title="Clean-keyword FcNN (dropout 0.5, val-loss stop)\ntest: features ≤ 2022 → impact 2023–2025", xlabel="FPR", ylabel="TPR")
ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "roc_cleankw_ir10_50_100.png"), dpi=130)
print("saved to", OUT)
