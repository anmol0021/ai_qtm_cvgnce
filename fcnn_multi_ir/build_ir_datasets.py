#!/usr/bin/env python3
"""
Balanced train / test sets for IR = 50 and IR = 100 (paper recipe, Appendix C):
all pairs with impact >= IR  +  an equal number of random pairs with impact < IR,
drawn from ALL unconnected pairs (full per-pair Parquet files from the feature pipeline).
  train: pairs_2019 (features 2017-2019, label 2020-2022)
  test : pairs_2022 (features 2020-2022, label 2023-2025)
IR = 10 reuses the existing balanced_ir10.csv / balanced_ir10_test.csv (built the same way).
Memory-light: impact column read first, rows then gathered part by part.

NOTE: needs the full per-pair Parquet files produced by the feature pipeline
(../feature_pipeline -> scripts/2_build_pair_features.py -> output/pairs_<year>/features_full).
Override the location with the PAIRS_DIR environment variable.
"""
import glob, os
import numpy as np, pandas as pd, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("PAIRS_DIR", os.path.join(HERE, "..", "feature_pipeline", "output"))
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


for ir in (50, 100):
    for year, name in ((2019, "train"), (2022, "test")):
        df, npos = build(year, ir, seed=42 + ir)
        path = os.path.join(OUT, f"balanced_ir{ir}_{name}.csv")
        df.to_csv(path, index=False)
        print(f"IR>={ir} {name} ({year}): {len(df)} rows, {npos} positives -> {path}")
