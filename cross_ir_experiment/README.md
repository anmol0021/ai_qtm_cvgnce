# Cross-IR experiment: replicating Gu & Krenn (2024), Fig. 8

**Paper's claim (Fig. 8):** an FcNN trained for IR = 10 does slightly better on the **IR = 50** task than an identical FcNN trained for IR = 50 itself (AUC 0.9416 vs 0.9365). The paper explains it as *"predictions for higher impact ranges benefit from training on more diverse data, including those from lower impact ranges."*

## Setup
- **Models:** the three FcNNs from `../fcnn_multi_ir`. They have identical architecture (141 → 4×600 → 1) and training parameters, and each was trained on its own balanced 2019 set. No retraining was done.
- **Test sets:** the balanced 2022 test sets (features ≤ 2022 → impact 2023–2025) for IR 10, 50 and 100. Every model scores every test set; AUC only needs the ranking of the scores.
- **Significance:** a paired bootstrap of the AUC difference (2,000 resamples of the test pairs).

Script: `cross_ir_eval.py`. Results: `results/`.

**To run:** first run `../fcnn_multi_ir/train_multi_ir.py`. It creates the three models (`results/ir*/fcnn_ir*.pt`), and the test CSVs must be in `../fcnn_multi_ir/data/`. Then run `python cross_ir_eval.py` (about 1 minute).

## Result 1: the paper's finding replicates
| Test task IR ≥ 50 (19,736 pairs) | AUC |
|---|---|
| FcNN trained on IR = 50 | 0.8137 |
| **FcNN trained on IR = 10** | **0.8271** |
| Difference (IR10 − IR50) | **+0.0133** (95% CI +0.0095 … +0.0169; IR10 better in 100% of bootstrap samples) |

The IR-10 model wins on the IR-50 task. Our gap (+0.013) is larger than the paper's (+0.005), and it is clearly outside test-sampling noise.

## Result 2: the full matrix adds nuance
| Model trained on ↓ / test task → | IR ≥ 10 | IR ≥ 50 | IR ≥ 100 |
|---|---|---|---|
| IR = 10 (74k training rows) | **0.860** | 0.827 | 0.813 |
| IR = 50 (17k training rows) | 0.838 | **0.814** | 0.795 |
| IR = 100 (6k training rows) | 0.840 | 0.831 | **0.825** |

1. **The IR-50 model is the weakest on every task, including its own.** Both the IR-10 and the IR-100 model beat it on the IR-50 task (0.827 and 0.831 vs 0.814).
2. **On the IR-100 task the effect reverses.** The IR-100 model is best (0.825); the IR-10 model is worse by −0.012 (CI −0.019 … −0.004).
3. So in our data, "training on the lower IR is better" holds for IR 50 but is not a general rule. The IR-50 result may partly reflect this one IR-50 training run being weak, rather than only the "more diverse data" effect.

## Caveats
- **Training variance is not covered:** each model is a single training run (seed 42). The bootstrap only covers test-sampling variance. Retraining the IR-10 and IR-50 models with several seeds would show whether the +0.013 gap is robust to training randomness.
- **Different setup from the paper:** the paper evaluates over several intervals (2014→2017 training, evaluation 2–5 years ahead) on about 21M papers. Ours is one interval (2019 → 2022) on 30k papers.

## Files
- `results/cross_ir_results.png`: Fig. 8-style ROC (IR ≥ 50 task), ROC on the IR ≥ 100 task, and the AUC matrix.
- `results/cross_ir_auc_matrix.csv`
- `results/cross_ir_results.json`: the matrix plus the bootstrap tests.
