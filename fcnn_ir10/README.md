# FcNN impact forecasting for AI × Quantum keyword pairs (IR = 10)

This folder trains a fully connected neural network and evaluates it. It follows Gu & Krenn (2024), *Forecasting high-impact research topics via machine learning on evolving knowledge graphs*, Appendix C.

**Task:** take a pair of research keywords that have never appeared together in a paper. Using only what was known up to the base year, predict whether, within the next **3 years**, new papers combining the pair will collect **≥ 10 citations** (impact range IR = 10).

## Contents
```
fcnn_ir10/
├── train_fcnn_ir10.py              ← the whole pipeline: load → preprocess → train → test → save
├── requirements.txt
├── data/
│   ├── balanced_ir10_train.csv     ← TRAIN: 87,272 pairs (features 2017–2019, label 2020–2022)
│   ├── balanced_ir10_test.csv      ← TEST : 86,316 pairs (features 2020–2022, label 2023–2025)
│   └── feature_dictionary.csv      ← the 141 feature names, groups and plain-English descriptions
└── results/                        ← output of our run (re-created when you run the script)
```

> **Datasets are not stored in this repo** (the IR-10 CSVs are 80–90 MB each). Get them by running `../feature_pipeline` (and, for IR 50/100, `build_ir_datasets.py`), or from the shared zip. Put them in `data/` with the file names shown above. `data/feature_dictionary.csv` is already here. Trained models (`*.pt`) and per-pair prediction files are also not committed; the script re-creates them.

## How to run
```
pip install -r requirements.txt
python train_fcnn_ir10.py
```
It runs on CPU in about 13 minutes and writes everything to `results/`.

## Data
Each row is one keyword pair *(u, v)* that is **unconnected in the base year y**: no paper up to year y mentions both keywords.

**Columns:**
- `u`, `v`, `keyword_u`, `keyword_v`
- 141 features, named `f_…`. They are the paper's Tables I–II: 20 node-network, 58 node-citation, 21 pair-network and 42 pair-citation features, computed for years y, y−1 and y−2. Descriptions are in `data/feature_dictionary.csv`.
- `impact_3y_citations`: total citations, up to the end of y+3, of the papers published in y+1 … y+3 that mention both keywords.
- `ir_ge_10`: **the label used here** (impact ≥ 10). `ir_ge_50` and `ir_ge_100` are also included.

**The two files:**

| File | Base year y | Features from | Label from papers in | Rows | Positives |
|---|---|---|---|---|---|
| train | 2019 | 2017–2019 | 2020–2022 | 87,272 | 43,641 (50%) |
| test | 2022 | 2020–2022 | 2023–2025 | 86,316 | 43,131 (50%) |

Both files are balanced the paper's way: all positive pairs plus an equal number of random negatives.

**Source:** a knowledge graph of 3,000 AI × Quantum keywords, built over 30,053 OpenAlex papers with OpenAlex yearly citation counts.

## Method
- **Split:** the train file is split 85% train / 15% validation (stratified, seed 42). The test file is used **whole** and is never seen during training.
- **Preprocessing:** `log1p` of every feature, then standardisation fitted on the training split only.
- **Model:** 141 → 600 → 600 → 600 → 600 → 1. That's 4 hidden layers × 600 neurons, as in the paper, with ReLU and dropout 0.2. There are **1,167,601 parameters**.
- **Training:**
  - Adam, learning rate 1e-4, weight decay 1e-5.
  - Batch size 1000, binary cross-entropy loss.
  - Up to 300 epochs, with early stopping (patience 25) on validation AUC. The best weights are restored at the end.

## Results of our run (test set, all 86,316 pairs)
| Metric | Value |
|---|---|
| **Test AUC** | **0.860** |
| Validation AUC | 0.864 |
| Accuracy | 0.785 |
| Precision | 0.795 |
| Recall | 0.768 |
| F1 | 0.781 |
| TPR at FPR = 0.1 / 0.3 | 0.636 / 0.842 |
| Confusion matrix | TN 34,620 · FP 8,565 · FN 10,001 · TP 33,130 |
| Epochs | 107 run, best epoch 82 (early stopping) |

**Top predictions:** of the 100 and of the 1,000 highest-scored test pairs, 99% really reached ≥ 10 citations.

**Reproducibility:** the seed is fixed (42). On a different CPU, thread count or torch version, results can differ in the third decimal, because floating-point summation order varies.

## Output files (`results/`)
- `metrics_ir10.json`: all metrics.
- `training_history_ir10.csv`: loss and validation AUC per epoch.
- `fcnn_ir10_curves.png`: loss curves, validation AUC and the test ROC curve.
- `test_predictions_ir10.csv`: every test pair with its predicted probability, sorted highest first.
- `fcnn_ir10.pt`: the trained weights, feature list and scaler (mean / std).
