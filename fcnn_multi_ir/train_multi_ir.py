#!/usr/bin/env python3
"""
FcNN trained and tested separately for IR = 10, 50, 100 (Gu & Krenn 2024, Fig. 4a / 11 style).
For each IR: train on the balanced 2019 set (85 % train / 15 % validation, early stopping on
validation AUC), test on the WHOLE balanced 2022 set of the same IR.
Identical model / hyper-parameters for every IR (same as ../fcnn_ir10):
  141 -> 600 -> 600 -> 600 -> 600 -> 1, ReLU, dropout 0.2; log1p + standardisation;
  Adam lr 1e-4, wd 1e-5, batch 1000, BCE, max 300 epochs, patience 25, seed 42.
"""
import json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
FEAT = pd.read_csv(os.path.join(DATA, "feature_dictionary.csv")).feature.tolist()
FILES = {
    ir: (os.path.join(DATA, f"balanced_ir{ir}_train.csv"), os.path.join(DATA, f"balanced_ir{ir}_test.csv"))
    for ir in (10, 50, 100)
}


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


def run(ir, seed=42, batch=1000, patience=25, max_epochs=int(os.environ.get("MAX_EPOCHS", 300))):
    torch.manual_seed(seed); np.random.seed(seed)
    label = f"ir_ge_{ir}"
    tr_all = pd.read_csv(FILES[ir][0]); te = pd.read_csv(FILES[ir][1])
    tr_all.columns = tr_all.columns.str.strip(); te.columns = te.columns.str.strip()
    tr, va = train_test_split(tr_all, test_size=0.15, stratify=tr_all[label], random_state=seed)
    prep = lambda d: np.log1p(d[FEAT].to_numpy(np.float64).clip(min=0))
    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr, Xva, Xte = [((x - mu) / sd).astype(np.float32) for x in (Xtr, Xva, Xte)]
    ytr, yva, yte = [d[label].to_numpy(np.float32) for d in (tr, va, te)]
    print(f"\n===== IR >= {ir}: train {len(tr)}  val {len(va)}  test {len(te)} =====", flush=True)

    model = FcNN(len(FEAT))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)

    def logits(X):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(X))

    hist, best, best_state, bad, t0 = [], -1, None, 0, time.time()
    for ep in range(1, max_epochs + 1):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad(); loss = lossf(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        lv = logits(Xva)
        va_loss = float(lossf(lv, torch.from_numpy(yva))); va_auc = roc_auc_score(yva, torch.sigmoid(lv).numpy())
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "val_loss": va_loss, "val_auc": va_auc})
        if va_auc > best + 1e-4:
            best, bad = va_auc, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 20 == 0 or ep == 1:
            print(f"  epoch {ep:3d} train_loss {tot/len(perm):.4f} val_loss {va_loss:.4f} val_auc {va_auc:.4f} [{time.time()-t0:.0f}s]", flush=True)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    p = torch.sigmoid(logits(Xte)).numpy(); yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
    fpr, tpr, _ = roc_curve(yte, p)
    top = pd.DataFrame({"p": p, "y": yte, "imp": te["impact_3y_citations"].to_numpy()}).sort_values("p", ascending=False)
    m = {"impact_range": ir, "train_rows": len(tr), "val_rows": len(va), "test_rows": len(te),
         "epochs_run": len(hist), "best_epoch": int(max(hist, key=lambda h: h["val_auc"])["epoch"]),
         "val_auc": round(best, 4), "test_auc": round(roc_auc_score(yte, p), 4),
         "test_accuracy": round(accuracy_score(yte, yhat), 4), "test_precision": round(precision_score(yte, yhat), 4),
         "test_recall": round(recall_score(yte, yhat), 4), "test_f1": round(f1_score(yte, yhat), 4),
         "TPR_at_FPR_0.1": round(float(np.interp(0.1, fpr, tpr)), 4),
         "TPR_at_FPR_0.3": round(float(np.interp(0.3, fpr, tpr)), 4),
         "top100_precision": round(float(top.y.head(100).mean()), 3),
         "top100_mean_impact": round(float(top.imp.head(100).mean()), 1),
         "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}
    print(json.dumps(m), flush=True)
    d = os.path.join(OUT, f"ir{ir}"); os.makedirs(d, exist_ok=True)
    json.dump(m, open(os.path.join(d, f"metrics_ir{ir}.json"), "w"), indent=2)
    pd.DataFrame(hist).to_csv(os.path.join(d, f"training_history_ir{ir}.csv"), index=False)
    o = te[["u", "v", "keyword_u", "keyword_v", "impact_3y_citations", label]].copy()
    o["predicted_prob"] = p; o["predicted_label"] = yhat
    o.sort_values("predicted_prob", ascending=False).to_csv(os.path.join(d, f"test_predictions_ir{ir}.csv"), index=False)
    torch.save({"state_dict": model.state_dict(), "features": FEAT, "mean": mu, "std": sd}, os.path.join(d, f"fcnn_ir{ir}.pt"))
    return m, pd.DataFrame(hist), (fpr, tpr)


res = {ir: run(ir) for ir in (10, 50, 100)}

# ------------------------------------------------------------------ combined figure
col = {10: "C0", 50: "C1", 100: "C2"}
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for ir, (m, h, (fpr, tpr)) in res.items():
    ax[0].plot(fpr, tpr, color=col[ir], lw=2, label=f"IR ≥ {ir}  (AUC = {m['test_auc']:.3f})")
    ax[1].plot(h.epoch, h.val_auc, color=col[ir], label=f"IR ≥ {ir}")
    ax[1].axvline(m["best_epoch"], color=col[ir], ls=":", lw=1)
    ax[2].plot(h.epoch, h.train_loss, color=col[ir], ls="-", label=f"IR ≥ {ir} train")
    ax[2].plot(h.epoch, h.val_loss, color=col[ir], ls="--", label=f"IR ≥ {ir} validation")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="random (AUC = 0.5)")
ax[0].set(title="ROC on test: features ≤2022 → impact 2023–2025", xlabel="False positive rate", ylabel="True positive rate")
ax[0].legend(loc="lower right")
ax[1].set(title="Validation AUC during training (2019 hold-out)", xlabel="epoch", ylabel="AUC"); ax[1].legend()
ax[2].set(title="Loss (BCE)", xlabel="epoch"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fcnn_ir10_50_100_curves.png"), dpi=130)

fig, a = plt.subplots(figsize=(6.2, 5.6))
for ir, (m, h, (fpr, tpr)) in res.items():
    a.plot(fpr, tpr, color=col[ir], lw=2, label=f"IR ≥ {ir}  (AUC = {m['test_auc']:.3f})")
a.plot([0, 1], [0, 1], "k--", lw=1, label="random (AUC = 0.5)")
a.set(title="FcNN ROC, test: features ≤2022 → impact 2023–2025", xlabel="False positive rate", ylabel="True positive rate")
a.legend(loc="lower right"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "roc_ir10_50_100.png"), dpi=130)

tab = pd.DataFrame([{k: v for k, v in r[0].items() if k != "confusion_matrix"} for r in res.values()])
tab.to_csv(os.path.join(OUT, "summary_ir10_50_100.csv"), index=False)
print("\n", tab.to_string(index=False))
