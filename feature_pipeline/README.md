# Keyword-pair feature pipeline: 141 features + impact labels (AI × Quantum)

This code builds an evolving, citation-augmented knowledge graph of AI × Quantum keywords. For every keyword pair it computes the **141 features** of Gu & Krenn (2024), *Forecasting high-impact research topics via machine learning on evolving knowledge graphs* (Appendix D, Tables I–II), plus the impact labels IR ≥ 10 / 50 / 100.

It produced `balanced_ir10.csv` (train, 2019) and `balanced_ir10_test.csv` (test, 2022), the files used by the FcNN models in `../fcnn_ir10` and `../fcnn_multi_ir`.

## Input data
| File | Where | Note |
|---|---|---|
| `ai_quantum_imputed.jsonl` | put it in `data/` (not included: 585 MB) | 30,053 OpenAlex papers. **The JSONL is required, not the CSV**, because only the JSONL has OpenAlex `counts_by_year` (citations per year), which the citation features need |
| `output/top3000_keywords.csv` | included | the 3,000 keywords (graph vertices) we used |
| `data/quantum_x_ai_300_anchors.xlsx.csv` | **not in the repo** (Web of Science export, not redistributable); put your copy in `data/` | 300 anchor papers; only needed to re-create the keywords |

## How to run
```
pip install -r requirements.txt
python scripts/1_extract_citations.py          # JSONL → output/_work/papers_citations.parquet   (~1–2 min)
python scripts/2_build_pair_features.py        # graph + 141 features + labels                     (~3 min)
python scripts/3_describe_features.py          # adds plain-English descriptions to the feature list
python scripts/4_sanity_check.py               # optional: data checks + quick AUC check
```
`1_extract_citations.py` also accepts the JSONL path as an argument: `python scripts/1_extract_citations.py path/to/ai_quantum_imputed.jsonl`.

Step 2 options:
- `--years 2019,2022`: base years.
- `--keywords <csv>`: any CSV with a `keyword` column.
- `--out <folder>`: where to write the outputs (default `output/`; citations are always read from `output/_work/`). Used by `../clean_keywords_experiment`.
- `--chunk 60`

This means you can rebuild the features for a different keyword list (e.g. a curated one) or different years.

## Outputs (`output/`)
```
pairs_2019/                         base year 2019: features 2017–2019, label from papers in 2020–2022
  features_full/part-*.parquet      ALL 4,361,106 pairs unconnected in 2019
  balanced_ir10.csv                 all IR≥10 positives + equal random negatives (87,272 rows) → TRAIN
pairs_2022/                         base year 2022: features 2020–2022, label from papers in 2023–2025
  features_full/part-*.parquet      ALL 4,258,739 pairs unconnected in 2022
  balanced_ir10.csv                 86,316 rows → TEST (we renamed this copy to balanced_ir10_test.csv)
feature_dictionary.csv              index 0–140, feature name, group, description
pairs_summary.json                  pair / positive counts per year
_work/                              intermediates (citation table, paper×keyword incidence matrix)
```
**Row layout:** `u, v, keyword_u, keyword_v`, the 141 `f_…` features, `impact_3y_citations`, `ir_ge_10`, `ir_ge_50`, `ir_ge_100`.

## What the code does
**1. Citations per paper (`1_extract_citations.py`).**
- Streams the JSONL and keeps, for each paper: id, publication year, title + abstract (the abstract is rebuilt from OpenAlex's `abstract_inverted_index`), `cited_by_count`, and citations per year for 2003–2026.

**2. Knowledge graph (`2_build_pair_features.py`).**
- **Concept detection:** a keyword is in a paper if it appears as a contiguous phrase in the title/abstract, with lower-casing and the last word singularised (*networks* = *network*). This gives a paper × keyword matrix **M** (30,053 × 3,000).
- **Graph up to year t:** keep the papers published ≤ t. Two keywords are connected if some remaining paper mentions both.
  - adjacency **A** = (MᵀM > 0)
  - degree **N** = row sums of A
  - shared neighbours = A·A
  - **PageRank** via networkx
- **Citation quantities** for each keyword at year t, all summed over the papers mentioning it:
  - **C** = citations received in year t
  - **Ct** = citations received up to t. For one paper, citations up to t = `cited_by_count` − (its citations after t).
  - **Ct3** = citations received in t−2 … t
  - **Pn** = number of papers
  - averages: Cm = C/Pn, Ctm = Ct/Pn, CtmΔ3 = Ct3/Pn

**3. Pairs and features.**
- All C(3000, 2) = 4,498,500 pairs are formed. Only pairs **unconnected in the base year y** are kept: the task is predicting new combinations.
- For each pair, the 141 features use the graph state at y, y−1 and y−2. Names, groups and descriptions are in `feature_dictionary.csv`.

| Index | Group | Features |
|---|---|---|
| 0–19 | node network (20) | degree, new neighbours over 1/2 years and their ranks, PageRank |
| 20–77 | node citation (58) | C, Ct, CtΔ3, Pn, Cm, Ctm, CtmΔ3; new citations / new papers over 1/2 years and their ranks |
| 78–98 | pair network (21) | shared neighbours, geometric, cosine, Simpson, preferential attachment, Sørensen–Dice, Jaccard |
| 99–140 | pair citation (42) | r1, r2, sums of Cm/Ctm/CtΔ3/CtmΔ3, min/max of C/Ct/CtΔ3/Pn |

**4. Label (prediction horizon = 3 years).**
- `impact_3y_citations` = the total citations, **up to the end of y+3**, of all papers published in **y+1 … y+3** that mention both keywords.
- `ir_ge_10/50/100` = impact ≥ 10 / 50 / 100.
- Note: a paper published in y+3 has at most 1 year to collect citations.

**5. Balanced file.** All IR ≥ 10 positives plus an equal number of random negatives (seed 42), as in the paper's training recipe (Appendix C).

**Memory:** pairs are processed in chunks of 60 first-keywords, each written to its own Parquet file. Peak memory is a few GB.

## Interpretations of the paper's Table I
- **Features 70–77:** the paper's formulas say "PR" but the description is "number of papers mentioning the node", so Pn is used.
- **Features 66–69:** rank of new **total** citations, as the description title says.
- **Features 12–13:** the formula has a typo (y−1 for u); the 2-year window is used for both nodes.
- **Ranks** are among all keywords, with 1 = largest and ties sharing the best rank.
- **PageRank** is on the unweighted graph.

## Optional: re-create the 3,000 keywords (`scripts/keyword_selection/`)
The keywords in `output/top3000_keywords.csv` came from:
1. `extract_keywords.py --input data/ai_quantum_imputed.csv --out output/keywords` (slow, about 2–3 h on CPU). This is the RAKE concept vocabulary of the corpus, which gives `output/keywords/concept_vocabulary.csv`. That file is already included.
2. `b_run_stratum_raw.py`: runs stratum-core's `tfidf_keybert.py` unmodified on the anchors. This is the raw baseline only.
3. `c_select_top3000_keywords.py`: RAKE candidates from the 300 anchors, scored with stratum-core's hybrid (0.9 SPECTER-KeyBERT + 0.1 TF-IDF, MMR 0.5), giving `output/top3000_keywords.csv`. It needs step 1 of the main pipeline first (`papers_citations.parquet`), and SPECTER downloads from Hugging Face.

stratum-core: https://github.com/Krishnakumarbj2004/stratum-core
