# Otomoto — Optimising an Artificial Neural Network for Marketing Segmentation

Module 6 assignment. Otomoto holds a large customer dataset but cannot segment its
audience effectively. The task is to optimise an artificial neural network so the
segmentation supports better marketing campaigns.

The brief refers to an "existing" ANN but supplies none, so the baseline is
constructed here explicitly, as directed by the course instructor. The graded work
is the optimisation applied on top of it.

## Start here

| Document | Purpose |
|---|---|
| **`REPORT.md`** | The main report. Process, algorithm justification, results, MSE reflection, recommendations, APA references. |
| **`DOCUMENTATION.md`** | Technical documentation: original model spec, algorithms applied, impact on segmentation. |
| **`SUBMISSION_CHECKLIST.md`** | Maps every brief requirement to where it is satisfied. **Read this before submitting** — it lists the items still needing your name and signature. |
| **`AI_DISCLOSURE.md`** | AI disclosure form. Needs your review and signature. |
| `README.md` | This file — orientation and quick results. |

## Data

`data/teleconnect.csv` — 7,043 customers, 21 columns, target `Churn`.

Three defects were found and handled in `src/data_prep.py`:

| Issue | Handling |
|---|---|
| `TotalCharges` stored as text with 11 blank strings | All 11 are `tenure = 0` customers who have never been billed, so 0 is the factually correct value. A mean fill would invent thousands of dollars of history for brand-new customers. |
| `customerID` is a unique identifier | Dropped. Left in, it contributes 7,043 meaningless dimensions after encoding. |
| Nine columns encode "No internet service" / "No phone service" as a third level | Kept as their own level. Structurally different from a plain "No" — the customer cannot have the add-on, rather than declining it. |

Class balance is 73.5% retained / 26.5% churned. That imbalance drives the metric
choices below.

Splits are 64/16/20 train/validation/test, stratified so the churn rate is
preserved in all three. The scaler is fitted on training data only.

## Framing

The dataset is telecom rather than automotive — a standard case-study mismatch.
It is treated as Otomoto's customer base, with churn risk as the signal that
drives segmentation. Predicting who is about to leave, and how much revenue they
represent, is a genuine marketing-segmentation problem.

## Method

Preprocessing is held identical across every model. The assignment measures the
effect of optimisation algorithms, so if the input representation moved between
runs an apparent "optimiser win" could really be a preprocessing win.

| Step | Script | What it does |
|---|---|---|
| 1 | `baseline.py` | Deliberately untuned reference: one hidden layer of 16 ReLU units, plain SGD at lr = 0.01, no regularisation, fixed 100 epochs, threshold 0.5 |
| 2 | `exp_optimizers.py` | **Experiment A** — all seven optimisers at one shared learning rate. **Experiment B** — full-batch vs mini-batch vs stochastic |
| 3 | `exp_tuning.py` | **Experiment C** — learning rate swept independently per optimiser (the fair comparison). **Experiment D** — cross-entropy vs MSE vs MAE. **Experiment E** — architecture and regularisation |
| 4 | `final_model.py` | Re-compares candidates at their own tuned thresholds, trains the winner, evaluates once on test against the baseline |
| 4b | `exp_final_evaluation.py` | Freezes the architecture and evaluates all seven optimisers on the test set (the brief asks for at least three) |
| 5 | `segmentation.py` | Converts risk scores into four costed marketing segments, validates them against actual churn, ranks the drivers |

### Metrics

Accuracy is reported but never relied on. At a 26.5% positive rate, predicting
"nobody churns" scores 0.7346 while being useless for a retention campaign.
Recall, F1, ROC-AUC and PR-AUC carry the real signal; log loss is the
cross-entropy the optimiser was actually minimising.

## Two methodological corrections

Both of these were found by running the experiments, and both changed the answer.

**1. Comparing optimisers at a shared learning rate is not a fair ranking.**
Experiment A held lr = 0.01 for every optimiser and plain SGD came first (0.8485)
with Adam last (0.8218). That rate suits SGD but is roughly ten times too large
for Adam, whose usual default is 0.001. Experiment C swept the rate per optimiser:
Adam moved from last to first (0.8517 at lr = 0.003), and the spread across all
seven collapsed from 0.0267 to 0.0037 AUC. Most of the apparent gap in Experiment
A was an artefact of the shared learning rate, not a property of the algorithms.

Experiment A is kept because it is a genuine controlled test, but it is labelled
as an unfair ranking and Experiment C is the one the conclusion rests on.

**2. Comparing F1 and recall at a fixed 0.5 threshold is the same confound
again.** Experiment E showed a class-weighted configuration reaching 0.809 recall
against 0.545 for the simple model. Class weighting shifts the calibration of the
output probabilities, so much of that gap was an artefact of where 0.5 happens to
fall. Given its own tuned threshold, the class-weighted model's recall fell to
0.682 and the simple 16-unit network won on validation F1. `final_model.py`
therefore retrains every candidate, tunes each one's threshold on validation, and
only then selects.

## Results

Held-out test set, 1,409 customers. Threshold tuned on validation only.

| Metric | Baseline | Optimised | Change |
|---|---|---|---|
| Accuracy | 0.8027 | 0.7722 | −3.8% |
| Precision | 0.6463 | 0.5588 | −13.6% |
| **Recall** | 0.5668 | **0.6738** | **+18.9%** |
| F1 | 0.6040 | 0.6109 | +1.1% |
| ROC-AUC | 0.8387 | 0.8446 | +0.7% |
| PR-AUC | 0.6299 | 0.6421 | +1.9% |
| Log loss | 0.4242 | 0.4186 | −1.3% |

In campaign terms: of 374 customers who actually churned, the baseline identified
212 and the optimised model identifies 252 — **40 more**, at the cost of 83 extra
false positives.

**Honest attribution.** ROC-AUC improved by only +0.0058. ROC-AUC is
threshold-free, so that small number is the true gain in the network's ability to
*rank* customers by risk — the part attributable to the optimiser, learning rate
and architecture work. The large recall gain comes mostly from moving the decision
threshold, which is a deployment choice rather than a better-trained network. Both
matter; reporting them as one number would overstate what the optimisation
achieved. Around 0.85 AUC is close to the practical ceiling for this dataset.

Accuracy and precision fell by design. A missed churner costs a subscription; a
false positive costs a discount offer.

### Findings from the experiments

- **Batch size** (Experiment B) illustrates the notes' taxonomy directly. At 60
  epochs, full-batch gradient descent made 60 weight updates and reached 0.7646
  AUC; mini-batch 32 made 8,460 and reached 0.8474. Same gradient, far fewer
  steps. True stochastic descent (batch = 1) cost 21× more per epoch for no gain.
- **Convergence speed** differs even where final scores converge. Momentum reached
  0.845 validation AUC in 4 epochs, plain SGD took 31, and AdaGrad 24.
- **Loss function** (Experiment D) confirmed cross-entropy over MSE for
  classification, which is the direct counterpart to the MSE reflection question.
- **More capacity did not help.** Every larger network (32-16, 64-32, with and
  without dropout, L2 and batch norm) scored below the 513-parameter model. With
  4,507 training rows and 30 features there is not enough signal to support more.

### All seven optimisers on the test set

Architecture frozen, each optimiser at its tuned learning rate and its own
validation-tuned threshold. Sorted by test F1.

| Model | lr | Threshold | Recall | F1 | ROC-AUC | Churners caught / 374 |
|---|---|---|---|---|---|---|
| Baseline (untuned SGD) | 0.01 | 0.50 | 0.5668 | 0.6040 | 0.8387 | 212 |
| Momentum (0.9) | 0.01 | 0.35 | **0.7139** | **0.6260** | 0.8441 | **267** |
| AdaGrad | 0.1 | 0.36 | 0.6898 | 0.6239 | 0.8438 | 258 |
| RMSProp | 0.001 | 0.38 | 0.6765 | 0.6209 | 0.8438 | 253 |
| SGD | 0.1 | 0.37 | 0.6631 | 0.6208 | 0.8445 | 248 |
| Nesterov (0.9) | 0.03 | 0.34 | 0.6925 | 0.6204 | 0.8431 | 259 |
| **Adam** (primary) | 0.003 | 0.39 | 0.6738 | 0.6109 | **0.8446** | 252 |
| Nadam | 0.003 | 0.38 | 0.6684 | 0.6061 | 0.8439 | 250 |

**All seven beat the baseline**, and the spread among them is only 0.0199 F1. Once an
optimiser has a suitable learning rate, which one you pick matters far less than the
fact that it was tuned at all.

Adam — the primary, chosen on validation — ranks 6th of 7 here. That is reported
rather than quietly swapped for Momentum: the choice was committed to on validation
evidence before these test numbers existed, so this ranking is an unbiased check. A
0.0199 spread on one 1,409-row split is within noise, which is why k-fold
cross-validation is the first recommendation.

### Segments

| Segment | Customers | Predicted risk | Actual churn | $/month | Tenure | Expected annual revenue at risk |
|---|---|---|---|---|---|---|
| Retain Now | 339 | 0.633 | 0.569 | $87.04 | 16.9 mo | $222,955 |
| Monitor | 112 | 0.562 | 0.527 | $47.33 | 5.6 mo | $36,099 |
| Grow | 366 | 0.152 | 0.150 | $92.02 | 53.6 mo | $62,058 |
| Nurture | 592 | 0.104 | 0.113 | $36.85 | 32.1 mo | $30,566 |

The segmentation holds up against reality: high-risk segments churned at 55.9%
versus 12.7% for low-risk, a 4.4× separation on data the model never saw.
Predicted risk tracks actual churn closely in all four groups.

Top drivers by permutation importance: `MonthlyCharges` (0.0716),
`Contract_Two year` (0.0607), `tenure` (0.0492), `Contract_One year` (0.0172).
Contract length is the actionable lever — moving month-to-month customers onto
longer terms attacks the strongest predictor directly.

## Running it

```bash
python -m pip install -r requirements.txt
python src/run_all.py          # full pipeline, ~35 min on CPU
```

Developed against Python 3.13.2, TensorFlow 2.21 and Keras 3.15 on CPU.
`requirements.txt` carries exact pins plus relaxed minimums if your Python differs.

Or step by step, in order (each depends on the previous one's output):

```bash
python src/baseline.py
python src/exp_optimizers.py
python src/exp_tuning.py              # longest step: 42 models in Experiment C
python src/final_model.py
python src/exp_final_evaluation.py
python src/segmentation.py
```

All RNGs are seeded (`RANDOM_SEED = 42` in `src/config.py`), so runs reproduce.

## Layout

```
REPORT.md                    main report
DOCUMENTATION.md             technical documentation
AI_DISCLOSURE.md             AI disclosure form (needs signature)
SUBMISSION_CHECKLIST.md      requirement-to-file mapping
data/teleconnect.csv         dataset
src/config.py                paths, seed, column groupings
src/data_prep.py             cleaning, encoding, stratified splits, scaling
src/modeling.py              model builder, optimiser factory, training, metrics
src/plots.py                 figure generation
src/baseline.py              step 1
src/exp_optimizers.py        step 2  (Experiments A, B)
src/exp_tuning.py            step 3  (Experiments C, D, E)
src/final_model.py           step 4
src/exp_final_evaluation.py  step 4b (all optimisers on test)
src/segmentation.py          step 5
src/run_all.py               runs all steps in order
results/tables/              20 CSV/JSON outputs
results/figures/             25 figures
models/                      baseline.keras, optimised.keras
```

## Limitations

- Value is proxied by `MonthlyCharges`. A real customer-lifetime-value model would
  fold in margin, acquisition cost and expected remaining tenure.
- The `Retain Now` segment is slightly over-predicted (0.633 predicted against
  0.569 actual). Probability calibration was not corrected; isotonic or Platt
  scaling would tighten it if the absolute probabilities were used for budgeting
  rather than the ranking.
- The 0.39 threshold maximises F1, which implicitly treats a false negative and a
  false positive as equally costly. They are not. With real figures for margin per
  customer and cost per retention offer, the threshold should be set to maximise
  expected profit instead.
- Single train/validation/test split. K-fold cross-validation would give tighter
  confidence in gaps as small as the 0.0037 AUC separating the tuned optimisers.
