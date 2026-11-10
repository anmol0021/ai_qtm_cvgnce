#!/usr/bin/env python3
"""
Balanced IR = 50 and IR = 100 train / test sets from the CLEAN-KEYWORD features
(../clean_keywords_experiment/features, built by ../feature_pipeline/scripts/2_build_pair_features.py
 --keywords keywords_quantum_ai_clean_final.csv --out ../clean_keywords_experiment/features).
Same recipe as ../fcnn_multi_ir/build_ir_datasets.py: all pairs with impact >= IR
+ an equal number of random pairs with impact < IR, drawn from ALL unconnected pairs.
  train: pairs_2019 (features 2017-2019, label 2020-2022)
  test : pairs_2022 (features 2020-2022, label 2023-2025)
"""
import glob, os
import numpy as np, pandas as pd, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "clean_keywords_experiment", "features")  # --out of 2_build_pair_features.py
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)


def build(year, ir, seed):
    parts = sorted(glob.glob(os.path.join(SRC, f"pairs_{year}", "features_full", "*.parquet")))
    imp = [pq.read_table(p, columns=["impact_3y_citations"]).column(0).to_numpy() for p in parts]
    sizes = np.array([len(x) for x in imp]); offs = np.concatenate([[0], sizes.cumsum()])
    allimp = np.concatenate(imp)
    pos = np.flatnonzero(allimp >= ir)
    neg = np.random.default_rng(seed).choice(np.flatnonzero(allimp < ir), size=len(pos), replace=False)
    keep = np.sort(np.concatenate([pos, neg]))
    out = []
    for k, p in enumerate(parts):
        local = keep[(keep >= offs[k]) & (keep < offs[k + 1])] - offs[k]
        if len(local):
            out.append(pq.read_table(p).to_pandas().iloc[local])
    df = pd.concat(out, ignore_index=True)
    assert (df[f"ir_ge_{ir}"] == 1).sum() == len(pos) == len(df) // 2
    return df, len(pos)


if __name__ == "__main__":
    for ir in (50, 100):
        for year, name in ((2019, "train"), (2022, "test")):
            df, npos = build(year, ir, seed=42 + ir)
            path = os.path.join(OUT, f"balanced_ir{ir}_{name}.csv")
            df.to_csv(path, index=False)
            print(f"IR>={ir} {name} ({year}): {len(df)} rows, {npos} positives -> {path}")
