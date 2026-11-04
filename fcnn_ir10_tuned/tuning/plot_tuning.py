import os, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for i, p in enumerate([0.0, 0.1, 0.2, 0.3, 0.5]):
    h = pd.read_csv(os.path.join(OUT, f"history_dropout{p}.csv"))
    ax[0].plot(h.epoch, h.val_auc, color=f"C{i}", label=f"dropout {p}")
    ax[1].plot(h.epoch, h.test_auc, color=f"C{i}", label=f"dropout {p}")
for a in ax: a.axhline(0.8598, color="k", ls=":", lw=1, label="reported model (test 0.8598)"); a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[0].set(title="Validation AUC (2019 hold-out) - used for selection", ylabel="AUC", ylim=(0.78, 0.885))
ax[1].set(title="Test AUC (2022 -> 2023-25) - shown only, never used")
ax[1].legend(loc="lower left"); plt.tight_layout(); plt.savefig(os.path.join(OUT, "tuning_curves.png"), dpi=130)
