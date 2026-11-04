# FcNN impact forecasting for AI × Quantum keyword pairs: IR = 10, 50, 100

**Task:** take a pair of research keywords that have never appeared together in a paper. Using only what was known up to the base year, predict whether, within the next **3 years**, new papers combining the pair will collect **≥ IR citations**. This follows Gu & Krenn (2024), *Forecasting high-impact research topics via machine learning on evolving knowledge graphs*.

## Contents
```
fcnn_multi_ir/
├── train_multi_ir.py         ← trains + tests one FcNN per IR, makes the comparison plots
├── build_ir_datasets.py      ← builds the IR 50/100 sets from the full pair Parquet files of ../feature_pipeline
├── requirements.txt
├── data/
│   ├── balanced_ir{10,50,100}_train.csv   ← 2019 pairs: features 2017–2019, label from papers in 2020–2022
│   ├── balanced_ir{10,50,100}_test.csv    ← 2022 pairs: features 2020–2022, label from papers in 2023–2025
│   └── feature_dictionary.csv             ← the 141 features (names, groups, descriptions)
└── results/                  ← output of our run (re-created when you run the script)
```

> **Datasets are not stored in this repo** (the IR-10 CSVs are 80–90 MB each). Get them by running `../feature_pipeline` (and, for IR 50/100, `build_ir_datasets.py`), or from the shared zip. Put them in `data/` with the file names shown above. `data/feature_dictionary.csv` is already here. Trained models (`*.pt`) and per-pair prediction files are also not committed; the script re-creates them.

## How to run
```
pip install -r requirements.txt
python train_multi_ir.py
```
It takes about 25 minutes on CPU (IR 10 about 15 min, IR 50 about 4 min, IR 100 about 2 min).
- Optional: set `MAX_EPOCHS=1` for a quick smoke test.
- The seed is fixed (42). Other CPUs or library versions can change results in the 3rd decimal.

**Columns in every CSV:** `u, v, keyword_u, keyword_v`, the 141 `f_…` features, `impact_3y_citations`, `ir_ge_10`, `ir_ge_50`, `ir_ge_100`.
- `impact_3y_citations` = the total citations, up to the end of y+3, of the papers published in y+1 … y+3 that mention both keywords.
- Each IR's model uses its own label column.


Each IR gets its own model, trained on 2019 pairs and tested on 2022 pairs with the **same** IR. All three use an identical model and hyper-parameters (same as the single-IR FcNN):
- architecture 141 → 4×600 (ReLU, dropout 0.2) → 1
- Adam lr 1e-4, batch 1000
- early stopping (patience 25) on validation AUC

## Data (`build_ir_datasets.py`)
Each IR uses the paper's recipe: **all pairs with impact ≥ IR, plus an equal number of random pairs with impact < IR**, drawn from all unconnected pairs.
- IR 10 uses the same files as `../fcnn_ir10`.
- IR 50 and 100 are built from the full per-pair Parquet files and saved in `data/`.

| IR | Train (2019 → 2020–22) | Validation | Test (2022 → 2023–25) |
|---|---|---|---|
| 10 | 74,181 | 13,091 | 86,316 |
| 50 | 17,117 | 3,021 | 19,736 |
| 100 | 6,446 | 1,138 | 6,256 |

## Results
| | IR ≥ 10 | IR ≥ 50 | IR ≥ 100 |
|---|---|---|---|
| **Test AUC** | **0.860** | **0.814** | **0.825** |
| Validation AUC (2019 hold-out) | 0.864 | 0.884 | 0.883 |
| Accuracy | 0.785 | 0.740 | 0.718 |
| Precision | 0.795 | 0.781 | 0.841 |
| Recall | 0.768 | 0.667 | 0.538 |
| F1 | 0.781 | 0.719 | 0.657 |
| TPR at FPR = 0.1 / 0.3 | 0.64 / 0.84 | 0.53 / 0.77 | 0.54 / 0.79 |
| Top-100 predicted: share ≥ IR | 99% | 98% | 94% |
| Top-100 predicted: mean impact | 48.9 | 89.4 | 145.8 |
| Epochs (best / run) | 82 / 107 | 120 / 145 | 125 / 150 |

The paper reports, for comparison, AUC 0.94 / 0.96 / 0.95 (Fig. 11), with 21M papers.

## How it behaves
1. **All three are far above random:** AUC 0.81–0.86. The top-100 predictions are 94–99% correct, and the top picks for IR 100 average 146 citations.
2. **Validation and test disagree for the higher IRs.**
   - Validation (2019) AUC is *higher* for IR 50/100 (0.88) than for IR 10 (0.86).
   - Test (2022) AUC is *lower* for IR 50/100 (0.81–0.82) than for IR 10 (0.86).
   - So the high-impact models generalise less well to the later period. The likely causes:
     - **Less training data:** 6–17k rows vs 74k. Training loss keeps falling while validation loss rises after about epoch 80–120 (overfitting).
     - **Temporal shift:** the field grew strongly in 2023–25, so citation-scale features at 2022 look different from 2019.
     - **Harder negatives:** for IR 50, pairs with 10–49 citations count as negatives.
3. **Recall drops as IR increases** (0.77 → 0.67 → 0.54), while precision stays high (0.78–0.84). At threshold 0.5 the higher-IR models are conservative on the 2022 data: when they say "high impact" they are usually right, but they miss many. That's a calibration shift, not a ranking failure; a lower threshold would trade precision for recall.
4. **Possible improvements:** train IR 50/100 on more data (e.g. stack base years 2018 + 2019), use smaller networks or more regularisation for the small sets, or recalibrate the threshold.

## Output files (`results/`)
- `fcnn_ir10_50_100_curves.png`: ROC (all three IRs), validation AUC per epoch, loss curves.
- `roc_ir10_50_100.png`: the ROC curves alone.
- `summary_ir10_50_100.csv`
- per IR (`ir10/`, `ir50/`, `ir100/`): metrics, training history, test predictions, model.
