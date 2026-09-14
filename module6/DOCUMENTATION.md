# Technical Documentation

**Otomoto ANN Optimization for Marketing Segmentation — Module 6**

Companion to `REPORT.md`. Where the report tells the story and argues the
conclusions, this document specifies the original model, the algorithms applied, and
their measured impact on segmentation, at a level of detail sufficient to audit or
rebuild the work.

---

## 1. The Original (Baseline) Model

### 1.1 Provenance

The assignment brief refers to an existing artificial neural network but supplies
none. Per the instructor's direction ("This baseline is not part of the assignment
brief so please assume and construct a simple NN model to be used as the baseline"),
the baseline was constructed here. It is intentionally untuned, representing a
plausible first-pass model written before optimization is considered.

Defined in `src/baseline.py` as `BASELINE_SPEC`. Saved artefact:
`models/baseline.keras`.

### 1.2 Full specification

```
Model: Sequential
  Input            (None, 30)
  Dense            (None, 16)   496 params    kernel 30x16 + bias 16
  Activation ReLU  (None, 16)     0 params
  Dense            (None, 1)     17 params    kernel 16x1 + bias 1
  Activation sigmoid

Total trainable parameters: 513
```

| Hyperparameter | Value | Status |
|---|---|---|
| Optimizer | SGD, `momentum = 0.0` | Textbook mini-batch gradient descent |
| Learning rate | 0.01 | Assumed, never tested |
| Loss | `binary_crossentropy` | Correct for the task; retained throughout |
| Batch size | 32 | Assumed |
| Epochs | 100 | Fixed, no early stopping |
| Weight init | Glorot uniform (Keras default) | Default |
| Dropout / L2 / BatchNorm | None | — |
| Class weighting | None | Imbalance unaddressed |
| Decision threshold | 0.50 | Default, untuned |

### 1.3 Measured performance

| Metric | Validation (1,127) | Test (1,409) |
|---|---|---|
| Accuracy | 0.8243 | 0.8027 |
| Precision | 0.7131 | 0.6463 |
| Recall | 0.5652 | 0.5668 |
| F1 | 0.6306 | 0.6040 |
| ROC-AUC | 0.8485 | 0.8387 |
| PR-AUC | 0.6643 | 0.6299 |
| Log loss | 0.4134 | 0.4242 |

Test confusion matrix: TP 212, FN 162, FP 116, TN 919.
Training time: 34.0 s, 100 epochs. Majority-class accuracy floor: 0.7346.

Raw output: `results/tables/baseline_results.json` (includes full per-epoch history).

### 1.4 Diagnosis driving the optimization plan

| Weakness | Evidence | Addressed by |
|---|---|---|
| Recall too low for campaign use | 0.5668 test; 162 of 374 churners missed | Threshold tuning; class weighting tested |
| Threshold unjustified | Fixed 0.50 on 73/27 data | Validation-tuned threshold (Step 4) |
| Learning rate arbitrary | 0.01 assumed | Experiment C, 6-point sweep per optimizer |
| No per-parameter adaptation | Plain SGD, one global step size | AdaGrad, RMSProp, Adam, Nadam |
| Slow convergence | 31 epochs to 0.845 val AUC | Momentum, Nesterov |
| Arbitrary stopping point | Fixed 100 epochs | Early stopping on val AUC, patience 15 |
| Imbalance unaddressed | No weighting | Class weighting tested (Experiment E) |
| Batch size untested | 32 assumed | Experiment B, 6 configurations |

Not a weakness: **overfitting**. Training and validation loss track closely
(`01_baseline_loss.png`), consistent with 513 parameters against 4,507 rows. This is
why regularization experiments were expected to yield little — and did not (§3.5).

---

## 2. Data Pipeline

Implemented in `src/data_prep.py`. Identical for every model in the study, so that
optimizer effects are not confounded with preprocessing effects.

### 2.1 Source

`data/teleconnect.csv` — 7,043 rows × 21 columns. Target `Churn` (Yes/No).
Distribution: 5,174 No (73.46%) / 1,869 Yes (26.54%). No missing values as loaded,
no duplicate rows, no duplicate `customerID`.

### 2.2 Cleaning

**`TotalCharges`** loads as `object` because 11 rows contain a single-space string.
Coerced with `pd.to_numeric(errors="coerce")`, then filled with `0.0`.

Justification: all 11 affected rows have `tenure == 0` — customers who joined but
have not been billed. Zero is factually correct. The code asserts this rather than
trusting it:

```python
assert (df.loc[never_billed, "tenure"] == 0).all()
```

If a future data refresh introduces a missing `TotalCharges` for a customer with
`tenure > 0`, the pipeline fails loudly instead of silently applying an assumption
that no longer holds.

**`customerID`** dropped — a unique identifier with no predictive content.

### 2.3 Encoding

| Group | Columns | Treatment |
|---|---|---|
| Target | `Churn` | `{No: 0, Yes: 1}` |
| Binary | `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | Mapped to 0/1 |
| Passthrough | `SeniorCitizen` | Already 0/1 |
| Continuous | `tenure`, `MonthlyCharges`, `TotalCharges` | `StandardScaler` |
| Multi-level | `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod` | One-hot, `drop_first=True` |

Nine service columns carry a third level ("No internet service" / "No phone
service"). These were **kept as distinct levels**, not collapsed into "No": a
customer who cannot hold an add-on is structurally different from one who declined
it. Permutation importance later confirmed these dummies carry real signal.

Resulting feature space: **30 dimensions**.

### 2.4 Splitting and scaling

```
train  4,507 rows  (26.54% churn)
val    1,127 rows  (26.53% churn)
test   1,409 rows  (26.54% churn)
```

Two successive stratified splits (`random_state = 42`) yield 64/16/20. Stratification
preserves the minority-class rate in all three; without it the churn rate can drift
between splits and the validation signal becomes unreliable.

`StandardScaler` is fitted on **training data only**, then applied to validation and
test. Fitting on the full dataset would leak test-set distributional information and
inflate reported scores. Only the three continuous features are scaled; one-hot
dummies are left at 0/1.

Scaling is not optional for this data. `TotalCharges` spans 0–8,684.8 while dummies
are 0/1; that disparity alone stalls plain gradient descent.

---

## 3. Optimization Algorithms Applied

Implemented in `src/modeling.py::make_optimizer`. All seven are Keras
implementations, selected to map onto the Module 5 material.

### 3.1 Algorithm reference

| Key | Class | Update rule | Selection rationale |
|---|---|---|---|
| `sgd` | `SGD(momentum=0.0)` | `w ← w − lr·g` | Control |
| `momentum` | `SGD(momentum=0.9)` | `v ← βv + g; w ← w − lr·v` | Damps oscillation in ravines; targets slow convergence |
| `nesterov` | `SGD(momentum=0.9, nesterov=True)` | Momentum with look-ahead gradient | Corrects overshoot before it occurs |
| `adagrad` | `Adagrad` | Step ÷ √(Σ g²) | Per-parameter rates for sparse one-hot features |
| `rmsprop` | `RMSprop` | Step ÷ √(EMA of g²) | Prevents AdaGrad's step decay to zero |
| `adam` | `Adam` | RMSProp + momentum, bias-corrected | Robust default combining both mechanisms |
| `nadam` | `Nadam` | Adam + Nesterov look-ahead | Tests whether look-ahead adds to Adam |

### 3.2 Reproducibility

`modeling.py::set_seeds` seeds Python `random`, NumPy, and `keras.utils.set_random_seed`
(covering TensorFlow) before every model build. Weight initialization, batch
shuffling and dropout masks are all stochastic; without seeding, a 0.004 AUC gap
between optimizers is indistinguishable from run-to-run noise.

Verified: the pipeline was executed end to end twice and final metrics reproduced
identically.

### 3.3 Experiment A — shared learning rate (`exp_optimizers.py`)

Constant: 30→16→1, lr 0.01, batch 32, 100 epochs, no regularization.
Output: `expA_optimizer_comparison.csv`, `expA_histories.json`.

| Optimizer | Val ROC-AUC | Val F1 | Val loss | Epochs → 0.845 AUC |
|---|---|---|---|---|
| SGD | 0.8485 | 0.6306 | 0.4134 | 31 |
| AdaGrad | 0.8484 | 0.6180 | 0.4137 | 24 |
| Momentum | 0.8449 | 0.5952 | 0.4188 | 4 |
| Nesterov | 0.8449 | 0.5982 | 0.4194 | 4 |
| Nadam | 0.8345 | 0.5458 | 0.4569 | 9 |
| RMSProp | 0.8294 | 0.5312 | 0.4586 | never |
| Adam | 0.8218 | 0.5947 | 0.4846 | never |

**This ranking is invalid as a statement about algorithm quality** — see §5.1. It is
retained because the contrast with Experiment C is instructive, and because the
convergence-speed column is valid: Momentum reaches the target in 4 epochs against
SGD's 31.

### 3.4 Experiment B — batch size (`exp_optimizers.py`)

Plain SGD, 60 epochs. Output: `expB_batch_size.csv`.

| Configuration | Batch | Updates/epoch | Total updates | Val ROC-AUC | s/epoch |
|---|---|---|---|---|---|
| Full-batch GD | 4,507 | 1 | 60 | 0.7646 | 0.12 |
| Mini-batch | 128 | 36 | 2,160 | 0.8409 | 0.18 |
| Mini-batch | 64 | 71 | 4,260 | 0.8449 | 0.23 |
| Mini-batch | 32 | 141 | 8,460 | 0.8474 | 0.33 |
| Mini-batch | 16 | 282 | 16,920 | 0.8489 | 0.63 |
| Stochastic | 1 | 4,507 | 13,521 (3 epochs) | 0.8449 | 6.96 |

Batch = 1 was run for 3 epochs only, to measure cost per update rather than to reach
convergence; at 4,507 updates per epoch it is 21× the per-epoch cost of batch 32.

### 3.5 Experiment C — learning rate per optimizer (`exp_tuning.py`)

7 optimizers × 6 learning rates = 42 models, max 100 epochs, early stopping
(patience 12). Output: `expC_optimizer_lr_grid.csv`,
`expC_lr_sensitivity_pivot.csv`, `expC_optimizer_best.csv`.

Best validation ROC-AUC:

| Optimizer | 0.0003 | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 | Best |
|---|---|---|---|---|---|---|---|
| SGD | 0.8233 | 0.8383 | 0.8449 | 0.8486 | 0.8496 | **0.8501** | 0.1 |
| Momentum | 0.8450 | 0.8485 | 0.8491 | **0.8500** | 0.8495 | 0.8408 | 0.01 |
| Nesterov | 0.8450 | 0.8486 | 0.8495 | 0.8498 | **0.8502** | 0.8415 | 0.03 |
| AdaGrad | 0.8035 | 0.8315 | 0.8429 | 0.8485 | 0.8486 | **0.8506** | 0.1 |
| RMSProp | 0.8464 | **0.8480** | 0.8454 | 0.8441 | 0.8383 | 0.8356 | 0.001 |
| Adam | 0.8472 | 0.8506 | **0.8517** | 0.8436 | 0.8404 | 0.8380 | 0.003 |
| Nadam | 0.8472 | 0.8489 | **0.8515** | 0.8476 | 0.8408 | 0.8445 | 0.003 |

Adam's optimum (0.003) and SGD's (0.1) sit at opposite ends of the grid. Spread once
tuned: **0.0037** ROC-AUC, against 0.0267 at the shared rate.

### 3.6 Experiment D — loss function (`exp_tuning.py`)

Adam at lr 0.003, 80 epochs. Output: `expD_loss_function.csv`.

| Loss | Val ROC-AUC | F1 | Recall | Precision | Comparable cross-entropy |
|---|---|---|---|---|---|
| `binary_crossentropy` | 0.8517 | 0.6008 | 0.5284 | 0.6960 | 0.4182 |
| `mse` | 0.8488 | 0.6011 | 0.5619 | 0.6462 | 0.4483 |
| `mae` | 0.8196 | 0.5898 | 0.5050 | 0.7089 | 2.1007 |

Cross-entropy is reported for all three regardless of what was minimized; raw loss
values are not comparable across different loss functions. See `REPORT.md` §7 for
the mechanism (sigmoid-saturation vanishing gradients under MSE).

### 3.7 Experiment E — architecture and regularization (`exp_tuning.py`)

Adam at lr 0.003, cross-entropy, early stopping on val AUC (patience 15), max 150
epochs. Output: `expE_architecture_regularisation.csv`, `expE_histories.json`.

| Configuration | Params | Val ROC-AUC | F1 @ 0.5 | Recall @ 0.5 | Stopped |
|---|---|---|---|---|---|
| 16 | 513 | **0.8515** | 0.6004 | 0.5452 | 45 |
| 64-32 + dropout 0.3 + batchnorm | 4,481 | 0.8485 | 0.6339 | 0.6254 | 18 |
| 32-16 + dropout 0.2 + class weight | 1,537 | 0.8476 | 0.6352 | 0.8094 | 19 |
| 64-32 + dropout 0.3 | 4,097 | 0.8468 | 0.6023 | 0.5318 | 23 |
| 32-16 | 1,537 | 0.8456 | 0.5765 | 0.4849 | 20 |
| 64-32 + L2 1e-4 | 4,097 | 0.8453 | 0.5574 | 0.4548 | 17 |
| 64-32 | 4,097 | 0.8449 | 0.5771 | 0.4883 | 19 |

Negative result worth recording: **no larger configuration beat 513 parameters.**
The F1/recall columns here are measured at a fixed 0.50 threshold and are **not
comparable across these rows** — see §5.2.

---

## 4. The Optimized Model

### 4.1 Selection procedure (`final_model.py::select_candidate`)

All seven Experiment E candidates were retrained, each given a threshold tuned on
validation by F1 maximization over 91 candidate thresholds in [0.05, 0.95], then
compared. Output: `final_candidate_selection.csv`.

| Configuration | Tuned t | Val F1 | Val recall | Val precision | Val ROC-AUC |
|---|---|---|---|---|---|
| **16** | 0.39 | **0.6614** | 0.6990 | 0.6276 | 0.8515 |
| 64-32 + dropout + batchnorm | 0.41 | 0.6499 | 0.7324 | 0.5840 | 0.8485 |
| 64-32 + dropout 0.3 | 0.40 | 0.6494 | 0.6722 | 0.6281 | 0.8468 |
| 64-32 + L2 1e-4 | 0.31 | 0.6436 | 0.6856 | 0.6065 | 0.8453 |
| 32-16 | 0.31 | 0.6428 | 0.7191 | 0.5811 | 0.8456 |
| 64-32 | 0.32 | 0.6406 | 0.6856 | 0.6012 | 0.8449 |
| 32-16 + dropout 0.2 + class weight | 0.64 | 0.6395 | 0.6823 | 0.6018 | 0.8476 |

Selection criterion: validation F1 at tuned threshold. Rationale: the business
objective is catching churners at a precision the campaign budget can absorb, not
maximizing a threshold-free ranking score. ROC-AUC is reported alongside as a check
that the choice is not winning on calibration alone — it is not; the selected model
also leads on ROC-AUC.

### 4.2 Final specification

Saved artefact: `models/optimised.keras`. Spec recorded in
`results/tables/final_results.json`.

```
Model: Sequential
  Input            (None, 30)
  Dense            (None, 16)   496 params
  Activation ReLU  (None, 16)
  Dense            (None, 1)     17 params
  Activation sigmoid

Total trainable parameters: 513
```

| Hyperparameter | Baseline | Optimized | Change rationale |
|---|---|---|---|
| Optimizer | SGD (momentum 0) | **Adam** | Per-parameter adaptation + momentum; best validation AUC at its tuned rate |
| Learning rate | 0.01 | **0.003** | Experiment C grid optimum for Adam |
| Loss | binary_crossentropy | binary_crossentropy | Unchanged — Experiment D confirmed it is correct |
| Architecture | 30→16→1 | 30→16→1 | Unchanged — Experiment E showed all larger nets worse |
| Batch size | 32 | 32 | Unchanged — Experiment B optimum region is 16–32 |
| Epochs | 100 fixed | **Early stopping, stopped at 45** | Monitors val AUC, patience 15, restores best weights |
| Regularization | None | None | Not needed; no overfitting observed |
| Class weighting | None | None | Tested; lost once thresholds were tuned fairly |
| **Threshold** | 0.50 | **0.39** | Tuned on validation; the single highest-impact change |

Note what did **not** change. Three of the eight rows are unchanged, and each is a
deliberate, evidence-backed decision to leave the baseline alone rather than an
omission.

### 4.3 Test-set performance

Output: `final_baseline_vs_optimised.csv`, `final_results.json`.

| Metric | Baseline | Optimized | Δ | Relative |
|---|---|---|---|---|
| Accuracy | 0.8027 | 0.7722 | −0.0305 | −3.8% |
| Precision | 0.6463 | 0.5588 | −0.0876 | −13.6% |
| Recall | 0.5668 | 0.6738 | **+0.1070** | **+18.9%** |
| F1 | 0.6040 | 0.6109 | +0.0069 | +1.1% |
| ROC-AUC | 0.8387 | 0.8446 | +0.0058 | +0.7% |
| PR-AUC | 0.6299 | 0.6421 | +0.0122 | +1.9% |
| Log loss | 0.4242 | 0.4186 | −0.0056 | −1.3% (better) |

| | TP | FN | FP | TN |
|---|---|---|---|---|
| Baseline | 212 | 162 | 116 | 919 |
| Optimized | 252 | 122 | 199 | 836 |

Also recorded at the default threshold for completeness
(`final_results.json::test_at_0.5`), so the threshold contribution can be isolated.

### 4.4 All optimizers on test (`exp_final_evaluation.py`)

Architecture frozen; only the optimizer varies, each at its Experiment C best rate
with its own validation-tuned threshold. Output:
`step4_optimizer_test_evaluation.csv`.

| Model | lr | t | Acc | Prec | Recall | F1 | ROC-AUC | Log loss | Caught/374 |
|---|---|---|---|---|---|---|---|---|---|
| BASELINE | 0.01 | 0.50 | 0.8027 | 0.6463 | 0.5668 | 0.6040 | 0.8387 | 0.4242 | 212 |
| Momentum | 0.01 | 0.35 | 0.7736 | 0.5574 | 0.7139 | **0.6260** | 0.8441 | 0.4193 | **267** |
| AdaGrad | 0.1 | 0.36 | 0.7793 | 0.5695 | 0.6898 | 0.6239 | 0.8438 | 0.4184 | 258 |
| RMSProp | 0.001 | 0.38 | 0.7807 | 0.5737 | 0.6765 | 0.6209 | 0.8438 | 0.4193 | 253 |
| SGD | 0.1 | 0.37 | 0.7850 | 0.5835 | 0.6631 | 0.6208 | 0.8445 | **0.4182** | 248 |
| Nesterov | 0.03 | 0.34 | 0.7750 | 0.5618 | 0.6925 | 0.6204 | 0.8431 | 0.4199 | 259 |
| Adam (primary) | 0.003 | 0.39 | 0.7722 | 0.5588 | 0.6738 | 0.6109 | **0.8446** | 0.4186 | 252 |
| Nadam | 0.003 | 0.38 | 0.7693 | 0.5543 | 0.6684 | 0.6061 | 0.8439 | 0.4187 | 250 |

7 of 7 beat the baseline F1. Test F1 spread: 0.0199.

**Multiple-comparison caveat.** Seven models scored on test means the best of seven
is optimistically biased. The primary was committed to on validation evidence before
these numbers existed; Adam ranks 6th of 7 here, which makes this an unbiased check
rather than a selection. Momentum's 0.6260 should not be reported as "the result".

---

## 5. Two Confounds Identified and Corrected

Documented because both produced confident but wrong conclusions, and both were
caught by asking whether a comparison was *fair* rather than merely *controlled*.

### 5.1 Shared learning rate invalidates optimizer ranking

**Symptom.** Experiment A ranked SGD first (0.8485), Adam last (0.8218).

**Diagnosis.** Every optimizer was forced to lr = 0.01. Adam's conventional default
is 0.001; at 0.01 its steps are ~10× too large and it oscillates rather than
settling. The experiment measured "which optimizer suits lr = 0.01", not which is
better.

**Correction.** Experiment C swept the rate per optimizer (42 models). Adam moved
from last to first (0.8517 at lr 0.003); spread fell 0.0267 → 0.0037. The
sensitivity pivot (§3.5) confirms opposite optima for Adam and SGD.

**Principle.** Optimizer and learning rate are not independent factors and cannot be
varied one at a time.

### 5.2 Fixed threshold invalidates cross-model recall comparison

**Symptom.** Experiment E showed 0.8094 recall for a class-weighted config against
0.5452 for the simple model — a gap large enough to drive the final choice.

**Diagnosis.** Both measured at a fixed 0.50 threshold. Class weighting shifts the
output probability distribution upward, so the weighted model crosses 0.50 far more
readily. Much of the gap was a calibration artefact, structurally identical to §5.1.

**Correction.** Per-candidate validation-tuned thresholds (§4.1). The weighted
model's recall fell 0.8094 → 0.6823 and it dropped to last on F1. Its ROC-AUC had
already signalled this (0.8476 vs 0.8515), being threshold-free and therefore immune.

**Principle.** Fixed-threshold metrics are not comparable across models with
different calibration. Use a threshold-free metric, or tune per model first.

---

## 6. Impact on Segmentation Results

Implemented in `src/segmentation.py`. This section answers the submission
requirement to document the algorithms' impact on segmentation specifically.

### 6.1 Construction

| Axis | Source | Cut-off |
|---|---|---|
| Risk | Optimized ANN predicted churn probability | 0.39 (validation-tuned operational threshold) |
| Value | `MonthlyCharges` | $69.95 (test-set median) |

Four quadrants: **Retain Now** (high/high), **Monitor** (high risk/low value),
**Grow** (low risk/high value), **Nurture** (low/low).

### 6.2 Segment profiles

Output: `segment_profiles.csv`, `scored_test_customers.csv` (per-customer scores).

| Segment | n | Pred risk | Actual churn | $/month | Tenure | Monthly rev | Annual rev | Expected annual at risk |
|---|---|---|---|---|---|---|---|---|
| Retain Now | 339 | 0.633 | 0.569 | $87.04 | 16.9 mo | $29,505 | $354,063 | **$222,955** |
| Monitor | 112 | 0.562 | 0.527 | $47.33 | 5.6 mo | $5,301 | $63,616 | $36,099 |
| Grow | 366 | 0.152 | 0.150 | $92.02 | 53.6 mo | $33,679 | $404,146 | $62,058 |
| Nurture | 592 | 0.104 | 0.113 | $36.85 | 32.1 mo | $21,816 | $261,790 | $30,566 |

Expected annual revenue at risk = Σ (per-customer churn probability ×
`MonthlyCharges` × 12). Using each customer's own probability rather than assuming
all flagged customers leave gives a materially more conservative figure.

### 6.3 Validation against actual outcomes

Segments were checked against realized churn on the held-out test set — data the
model never saw during training or selection.

| Group | Actual churn rate |
|---|---|
| High-risk segments (Retain Now + Monitor) | **55.9%** |
| Low-risk segments (Grow + Nurture) | **12.7%** |
| Separation | **4.4×** |

Predicted risk tracks actual churn in all four segments: 0.633/0.569, 0.562/0.527,
0.152/0.150, 0.104/0.113. Slight over-prediction in the two high-risk segments;
ranking is unaffected but absolute probabilities would benefit from calibration
before being used for budget forecasting.

### 6.4 Segmentation quality: before and after

The direct impact of the optimization on segmentation is the change in how many real
churners land in the actionable high-risk segments.

| | Baseline (t = 0.50) | Optimized (t = 0.39) |
|---|---|---|
| Churners correctly placed in high-risk segments | 212 of 374 | **252 of 374** |
| Churners missed (placed in low-risk segments) | 162 | **122** |
| Retained customers incorrectly in high-risk | 116 | 199 |
| Share of all churners reached by campaign | 56.7% | **67.4%** |

The optimization moved 40 additional real churners into the population a retention
campaign will contact — a 10.7 percentage-point increase in campaign coverage of the
at-risk base — at the cost of 83 additional wasted contacts.

### 6.5 Risk drivers (permutation importance)

Method: shuffle one feature column, measure the drop in test ROC-AUC, average over 3
repeats. A neural network exposes no readable coefficients, so this is the practical
route to interpretability. Output: `feature_importance.csv`.

| Rank | Feature | ROC-AUC drop |
|---|---|---|
| 1 | `MonthlyCharges` | 0.0716 |
| 2 | `Contract_Two year` | 0.0607 |
| 3 | `tenure` | 0.0492 |
| 4 | `Contract_One year` | 0.0172 |

Contract length ranks 2nd and 4th; tenure 3rd. This confirms the AdaGrad selection
rationale in §3.1 — the sparse one-hot contract dummies really are among the
strongest predictors, which is precisely the case for per-parameter adaptive step
sizes.

Caveat: permutation importance assumes feature independence. `MonthlyCharges` and
`TotalCharges` are correlated, so both may be understated as the model can partially
recover shuffled information from the correlate.

### 6.6 Campaign playbook

| Segment | n | Action | Budget |
|---|---|---|---|
| Retain Now | 339 | Personal outreach; targeted contract-upgrade or loyalty discount | Highest per customer |
| Monitor | 112 | Automated email offers, self-service nudges; no agent time | Minimal |
| Grow | 366 | Cross-sell / upsell — **not** discount | Moderate, revenue-generating |
| Nurture | 592 | Low-touch engagement, organic upgrade paths | Lowest |

The `Grow` segment is the counter-intuitive one: highest revenue ($92.02/month,
$404,146 annually) but only 15.0% actual churn. Blanket discounting would give away
margin on 366 customers who were not leaving.

---

## 7. Code Map and Reproduction

```
data/teleconnect.csv              supplied dataset
src/config.py                     paths, RANDOM_SEED = 42, column groupings
src/data_prep.py                  cleaning, encoding, stratified splits, scaling
src/modeling.py                   ModelSpec, optimizer factory, training, metrics
src/plots.py                      figure generation (Agg backend)
src/baseline.py                   Step 1
src/exp_optimizers.py             Step 3 — Experiments A, B
src/exp_tuning.py                 Step 3 — Experiments C, D, E
src/final_model.py                Step 4 — selection, final model, comparison
src/exp_final_evaluation.py       Step 4b — all optimizers on test
src/segmentation.py               Step 5 — segments, validation, importance
src/run_all.py                    runs steps 1-5 in order
results/tables/                   20 CSV/JSON outputs
results/figures/                  25 PNG figures
models/baseline.keras             original model
models/optimised.keras            optimized model
```

### 7.1 Environment

Python 3.13.2, TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.9.1, pandas 2.3.3,
NumPy 2.4.0, matplotlib 3.11.2, seaborn 0.13.2. CPU only (TensorFlow ≥ 2.11 has no
native GPU support on Windows).

### 7.2 Running

```bash
python src/run_all.py                  # steps 1-5, ~35 min CPU
python src/exp_final_evaluation.py     # step 4b, ~3 min
```

Or individually, in order — each step consumes files written by the previous one:

```bash
python src/baseline.py
python src/exp_optimizers.py
python src/exp_tuning.py               # longest: 42 models in Experiment C
python src/final_model.py
python src/exp_final_evaluation.py
python src/segmentation.py
```

### 7.3 Verification

All RNGs seeded; the pipeline was run end to end twice with identical final metrics.
Every figure quoted in `REPORT.md` and this document traces to a file in
`results/tables/`. The test set was used only after model selection concluded on
validation data.

### 7.4 Output inventory

| File | Contents |
|---|---|
| `baseline_results.json` | Baseline spec, val/test metrics, full training history |
| `expA_optimizer_comparison.csv` | Experiment A, 7 optimizers at shared lr |
| `expA_histories.json` | Per-epoch histories for Experiment A |
| `expB_batch_size.csv` | Experiment B, 6 batch configurations |
| `expC_optimizer_lr_grid.csv` | Experiment C, all 42 runs |
| `expC_lr_sensitivity_pivot.csv` | Optimizer × learning rate pivot |
| `expC_optimizer_best.csv` | Best lr per optimizer |
| `expD_loss_function.csv` | Experiment D, 3 loss functions |
| `expE_architecture_regularisation.csv` | Experiment E, 7 configurations |
| `expE_histories.json` | Per-epoch histories for Experiment E |
| `tuning_selection.json` | Selected optimizer, lr, loss, full winning spec |
| `final_candidate_selection.csv` | Candidates at their own tuned thresholds |
| `final_baseline_vs_optimised.csv` | Headline before/after comparison |
| `final_results.json` | Final spec, threshold, val/test metrics, history |
| `step4_optimizer_test_evaluation.csv` | All optimizers on test |
| `step4_optimizer_test_evaluation.json` | Same, with frozen architecture record |
| `segment_profiles.csv` | Four segments with profiles and revenue at risk |
| `scored_test_customers.csv` | Per-customer probability, segment, actual outcome |
| `feature_importance.csv` | Permutation importance, all 30 features |
| `segmentation_results.json` | Segment summary, validation, playbook |
