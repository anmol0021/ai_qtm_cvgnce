#!/usr/bin/env python3
"""
IR = 10 FcNN on features built with the peer's clean keyword list
(keywords_quantum_ai_clean_final.csv, 3,973 keywords). Identical to ../fcnn_ir10_tuned/train_fcnn_ir10_tuned.py
(dropout 0.5 + early stopping on validation LOSS, patience 10); only the input data differ.

Train : data/balanced_ir10_train.csv  (base year 2019: features 2017-2019, label = >=10 citations
        from new papers in 2020-2022); split 85 % train / 15 % validation
Test  : data/balanced_ir10_test.csv   (base year 2022: features 2020-2022, label from 2023-2025)
        Both produced by ../feature_pipeline/scripts/2_build_pair_features.py --keywords <clean list>

Model : 141 -> 600 -> 600 -> 600 -> 600 -> 1  (paper: 4 hidden layers x 600 neurons)
        ReLU, dropout 0.5
Input : log1p(feature) then standardisation fitted on the training split only
        (citation / degree features are heavy-tailed, spanning 0 .. 10^5)
Optim : Adam lr 1e-4, weight decay 1e-5, batch 1000 (paper: Adam, batch 1000, lr 3e-5),
        BCE loss, max 300 epochs, early stopping patience 10 on validation loss (min_delta 1e-4),
        restore the epoch with the lowest validation loss
"""
import json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")  # balanced_ir10_train/test.csv built with the clean keywords
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
LABEL = "ir_ge_10"

# ---------------------------------------------------------------- data
feat = pd.read_csv(os.path.join(DATA, "feature_dictionary.csv")).feature.tolist()
tr_all = pd.read_csv(os.path.join(DATA, "balanced_ir10_train.csv"))
te_all = pd.read_csv(os.path.join(DATA, "balanced_ir10_test.csv"))
tr_all.columns = tr_all.columns.str.strip(); te_all.columns = te_all.columns.str.strip()

te = te_all  # whole test file (previous run used a 25% sample)
tr, va = train_test_split(tr_all, test_size=0.15, stratify=tr_all[LABEL], random_state=SEED)
print(f"train {len(tr)}  val {len(va)}  test {len(te)} (whole test file)  | "
      f"positive share train {tr[LABEL].mean():.3f} test {te[LABEL].mean():.3f}")

def prep(df):
    return np.log1p(df[feat].to_numpy(np.float64).clip(min=0))
Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
Xtr, Xva, Xte = [((x - mu) / sd).astype(np.float32) for x in (Xtr, Xva, Xte)]
ytr, yva, yte = [d[LABEL].to_numpy(np.float32) for d in (tr, va, te)]

# ---------------------------------------------------------------- model
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

DROPOUT = 0.5
model = FcNN(len(feat), p=DROPOUT)
n_params = sum(p.numel() for p in model.parameters())
print(model, f"\ntrainable parameters: {n_params:,}")
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
lossf = nn.BCEWithLogitsLoss()
Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)

def predict(X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()

hist, best_loss, ref_loss, best_state, bad, PATIENCE, BATCH = [], np.inf, np.inf, None, 0, 10, 1000
t0 = time.time()
for epoch in range(1, 301):
    model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0.0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad(); loss = lossf(model(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        lva = model(torch.from_numpy(Xva))
    va_loss = float(lossf(lva, torch.from_numpy(yva))); va_auc = roc_auc_score(yva, lva.numpy())
    hist.append({"epoch": epoch, "train_loss": tot / len(perm), "val_loss": va_loss, "val_auc": va_auc})
    if va_loss < best_loss:                      # keep the lowest-validation-loss weights
        best_loss, best_ep, best_auc = va_loss, epoch, va_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if va_loss < ref_loss - 1e-4:                # patience counter: improvement > min_delta
        ref_loss, bad = va_loss, 0
    else:
        bad += 1
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch:3d}  train_loss {tot/len(perm):.4f}  val_loss {va_loss:.4f}  val_auc {va_auc:.4f}  [{time.time()-t0:.0f}s]", flush=True)
    if bad >= PATIENCE:
        print(f"early stop at epoch {epoch}; restoring epoch {best_ep} (val loss {best_loss:.4f})"); break
model.load_state_dict(best_state)

# ---------------------------------------------------------------- test
pte = predict(Xte); yhat = (pte >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
m = {"impact_range": 10, "train_rows": len(tr), "val_rows": len(va), "test_rows": len(te),
     "trainable_parameters": n_params, "epochs_run": len(hist),
     "dropout": DROPOUT, "early_stopping": "val_loss, patience 10", "best_epoch": int(best_ep),
     "val_loss": round(best_loss, 4),
     "val_auc": round(best_auc, 4), "test_auc": round(roc_auc_score(yte, pte), 4),
     "test_accuracy": round(accuracy_score(yte, yhat), 4),
     "test_precision": round(precision_score(yte, yhat), 4), "test_recall": round(recall_score(yte, yhat), 4),
     "test_f1": round(f1_score(yte, yhat), 4),
     "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}}
fpr, tpr, _ = roc_curve(yte, pte)
for f in (0.1, 0.3):
    m[f"test_TPR_at_FPR_{f}"] = round(float(np.interp(f, fpr, tpr)), 4)
print(json.dumps(m, indent=2))

json.dump(m, open(os.path.join(OUT, "metrics_ir10_cleankw.json"), "w"), indent=2)
pd.DataFrame(hist).to_csv(os.path.join(OUT, "training_history_ir10_cleankw.csv"), index=False)
te_out = te[["u", "v", "keyword_u", "keyword_v", "impact_3y_citations", LABEL]].copy()
te_out["predicted_prob"] = pte; te_out["predicted_label"] = yhat
te_out.sort_values("predicted_prob", ascending=False).to_csv(os.path.join(OUT, "test_predictions_ir10_cleankw.csv"), index=False)
torch.save({"state_dict": model.state_dict(), "features": feat, "mean": mu, "std": sd,
            "hidden": (600, 600, 600, 600), "dropout": DROPOUT}, os.path.join(OUT, "fcnn_ir10_cleankw.pt"))

# ---------------------------------------------------------------- plots
OLD = os.path.join(HERE, "..", "fcnn_ir10_tuned", "results", "test_predictions_ir10_tuned.csv")  # optional comparison
old = pd.read_csv(OLD) if os.path.exists(OLD) else None
if old is not None:
    fo, to, _ = roc_curve(old[LABEL], old.predicted_prob); old_auc = roc_auc_score(old[LABEL], old.predicted_prob)
H = pd.DataFrame(hist)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].plot(H.epoch, H.train_loss, label="train"); ax[0].plot(H.epoch, H.val_loss, label="validation")
ax[0].axvline(best_ep, ls="--", color="grey", label=f"restored epoch {best_ep}")
ax[0].set(title="Loss (BCE), dropout 0.5", xlabel="epoch"); ax[0].legend()
ax[1].plot(H.epoch, H.val_auc, color="C2"); ax[1].axvline(best_ep, ls="--", color="grey")
ax[1].set(title="Validation AUC (2019 hold-out)", xlabel="epoch")
ax[2].plot(fpr, tpr, color="C3", lw=2, label=f"peer clean keywords  AUC = {m['test_auc']:.3f}")
if old is not None: ax[2].plot(fo, to, color="C0", lw=1.5, alpha=.8, label=f"our top-3000 keywords (own test set)  AUC = {old_auc:.3f}")
ax[2].plot([0, 1], [0, 1], "k--", lw=1, label="random 0.5")
ax[2].set(title="ROC on test, IR = 10 (2022 -> impact 2023-25)", xlabel="FPR", ylabel="TPR")
ax[2].legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fcnn_ir10_cleankw_curves.png"), dpi=130)
print("saved to", OUT)
