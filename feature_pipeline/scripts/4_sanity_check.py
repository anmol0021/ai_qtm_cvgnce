"""Step 5: sanity checks + quick predictive validation in the paper's setup
(train on y=2019 -> labels 2020-22, evaluate on y=2022 -> labels 2023-25; AUC per IR)."""
import glob, os, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
feat = pd.read_csv(os.path.join(OUT, "feature_dictionary.csv")).feature.tolist()
tr = pd.read_csv(os.path.join(OUT, "pairs_2019", "balanced_ir10.csv"))
print("balanced 2019:", tr.shape, "positives:", int(tr.ir_ge_10.sum()), "| NaN:", int(tr[feat].isna().sum().sum()),
      "| negative values:", int((tr[feat] < 0).sum().sum()))
rng = np.random.default_rng(0)
parts = sorted(glob.glob(os.path.join(OUT, "pairs_2022", "features_full", "*.parquet")))
ev = pd.concat([pq.read_table(p).to_pandas().sample(frac=0.15, random_state=1) for p in parts])
print("eval sample 2022:", ev.shape)
res = {}
for ir in [10, 50, 100]:
    y = (tr.impact_3y_citations >= ir).astype(int)
    # balance per IR like the paper (all positives + equal negatives)
    pos = tr[y == 1]; neg = tr[y == 0].sample(n=min(len(pos), int((y == 0).sum())), random_state=0)
    d = pd.concat([pos, neg]); yy = (d.impact_3y_citations >= ir).astype(int)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(d[feat], yy)
    p = m.predict_proba(ev[feat])[:, 1]
    res[f"IR>={ir}"] = round(roc_auc_score(ev[f"ir_ge_{ir}"], p), 4)
    top = ev.assign(p=p).nlargest(100, "p")
    print(f"IR>={ir}: eval AUC={res[f'IR>={ir}']}  | top-100 predicted pairs mean impact="
          f"{top.impact_3y_citations.mean():.1f} vs all={ev.impact_3y_citations.mean():.3f}")
    if ir == 100:
        print(top.head(10)[["keyword_u", "keyword_v", "impact_3y_citations"]].to_string(index=False))
json.dump(res, open(os.path.join(OUT, "validation_auc.json"), "w"), indent=2)
