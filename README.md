# AI × Quantum convergence: forecasting high-impact research combinations

We apply Gu & Krenn (2024), *Forecasting high-impact research topics via machine learning on evolving knowledge graphs* ([arXiv:2402.08640](https://arxiv.org/abs/2402.08640)), to the convergence of **artificial intelligence and quantum technologies**.

**Question:** given two research keywords that have never appeared together in a paper, will a paper combining them collect ≥ IR citations within the next 3 years?

## Pipeline
```
OpenAlex corpus (30,053 AI × Quantum papers, 2003–2025)
        │
        ▼
feature_pipeline/        knowledge graph of 3,000 keywords → 141 features per unconnected keyword pair
        │                + impact labels IR ≥ 10 / 50 / 100       (train: base year 2019, test: base year 2022)
        ▼
fcnn_ir10/               fully connected NN (4 × 600) for IR ≥ 10            → test AUC 0.860
fcnn_multi_ir/           the same NN trained and tested for IR ≥ 10, 50, 100 → test AUC 0.860 / 0.814 / 0.825
```

| Folder | What it does | Main script |
|---|---|---|
| [`feature_pipeline/`](feature_pipeline) | Extracts yearly citations, builds the evolving keyword graph, computes the paper's **141 features** (Tables I–II: 20 node-network, 58 node-citation, 21 pair-network, 42 pair-citation) and the labels. `scripts/keyword_selection/` shows how the 3,000 keywords were chosen (RAKE + stratum-core's KeyBERT/TF-IDF scorer). | `scripts/2_build_pair_features.py` |
| [`fcnn_ir10/`](fcnn_ir10) | Trains on 2019 pairs (label from 2020–2022) and tests on all 2022 pairs (label from 2023–2025) at IR = 10 | `train_fcnn_ir10.py` |
| [`fcnn_multi_ir/`](fcnn_multi_ir) | One model per IR = 10, 50, 100, plus the combined ROC plot | `train_multi_ir.py` |

Each folder has its own README with details, how to run it, and results.

## Results (test: features ≤ 2022 → impact 2023–2025)
| | IR ≥ 10 | IR ≥ 50 | IR ≥ 100 |
|---|---|---|---|
| Test AUC | **0.860** | **0.814** | **0.825** |
| Top-100 predicted pairs that really reached the IR | 99% | 98% | 94% |

![ROC IR 10/50/100](fcnn_multi_ir/results/roc_ir10_50_100.png)

## Quick start
```
pip install -r requirements.txt

# 1) features (needs data/ai_quantum_imputed.jsonl in feature_pipeline/data/)
cd feature_pipeline
python scripts/1_extract_citations.py
python scripts/2_build_pair_features.py
python scripts/3_describe_features.py

# 2) models: copy the balanced CSVs into each model's data/ folder:
#    output/pairs_2019/balanced_ir10.csv → balanced_ir10_train.csv
#    output/pairs_2022/balanced_ir10.csv → balanced_ir10_test.csv
cd ../fcnn_ir10 && python train_fcnn_ir10.py
cd ../fcnn_multi_ir && python build_ir_datasets.py && python train_multi_ir.py
```

## Data not in this repo
- **`ai_quantum_imputed.jsonl`** (585 MB): the OpenAlex corpus with `counts_by_year`.
- **Generated pair datasets:** CSVs of 80–90 MB, and about 1 GB of Parquet.
- **Trained model weights.**
- **The 300 anchor papers file:** it's a Web of Science export and can't be redistributed.

The repo does include the 3,000 keywords (`feature_pipeline/output/top3000_keywords.csv`) and the feature dictionary.

## Credits
- Method: X. Gu & M. Krenn, *Forecasting high-impact research topics via machine learning on evolving knowledge graphs* (Impact4Cast).
- Keyword scoring: [stratum-core](https://github.com/Krishnakumarbj2004/stratum-core) (`tfidf_keybert.py`, used unmodified).
- Data: [OpenAlex](https://openalex.org).
