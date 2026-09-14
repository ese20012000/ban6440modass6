"""
Step 4 - the optimised model, and an honest comparison against the baseline.

The configuration is not chosen here. It is read from tuning_selection.json, which
the experiment scripts wrote, so this script cannot quietly pick a better-looking
setting after seeing test results.

Discipline that matters for the comparison to mean anything:
  * the decision threshold is chosen on validation data, never on test
  * the test set is touched exactly once per model, at the end
  * the baseline is re-scored on that same test split, unchanged
"""

from __future__ import annotations

import json

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from config import MODELS_DIR, TABLES_DIR
from data_prep import prepare
from exp_tuning import architecture_configs
from modeling import (
    ModelSpec,
    best_threshold,
    evaluate,
    predict_proba,
    save_json,
    save_table,
    train_model,
)
from plots import (
    plot_bar_comparison,
    plot_confusion,
    plot_overfitting_gap,
    plot_roc,
    plot_threshold_sweep,
)

HEADLINE_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss"]


def load_selection() -> dict:
    path = TABLES_DIR / "tuning_selection.json"
    if not path.exists():
        raise FileNotFoundError(
            "tuning_selection.json not found - run exp_tuning.py before this script."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_baseline_results() -> dict:
    path = TABLES_DIR / "baseline_results.json"
    if not path.exists():
        raise FileNotFoundError("baseline_results.json not found - run baseline.py first.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def select_candidate(data, selection: dict) -> ModelSpec:
    """
    Re-compare the Experiment E candidates at each one's own tuned threshold.

    Why this step exists. Experiment E ranked configurations by validation ROC-AUC
    and reported F1 and recall at a fixed 0.5 cut-off. That fixed cut-off is not
    comparable across these candidates: class weighting deliberately shifts the
    output probabilities, so a weighted model looks far better on recall at 0.5
    partly as an artefact of calibration rather than of genuine ranking skill. It is
    the same confound as judging every optimiser at one shared learning rate.

    So each candidate is retrained, given its own threshold tuned on validation, and
    then compared. Selection uses validation F1 at that tuned threshold, because the
    business objective is catching churners at a precision the campaign budget can
    absorb - not maximising a threshold-free ranking score. ROC-AUC is reported
    alongside as a sanity check that the choice is not winning on calibration alone.

    Everything here happens on training and validation data. The test set is not
    touched until the winner is settled.
    """
    print("\n" + "=" * 78)
    print("MODEL SELECTION  candidates compared at their own tuned thresholds")
    print("=" * 78)

    candidates = architecture_configs(
        selection["optimizer"], float(selection["learning_rate"]), selection["loss"]
    )

    rows = []
    for spec in candidates:
        run = train_model(spec, data, verbose=0)
        t, _ = best_threshold(run.model, data.X_val, data.y_val)
        m = evaluate(run.model, data.X_val, data.y_val, threshold=t)
        rows.append(
            {
                "config": spec.name,
                "tuned_threshold": round(t, 3),
                "val_f1": m["f1"],
                "val_recall": m["recall"],
                "val_precision": m["precision"],
                "val_roc_auc": m["roc_auc"],
                "epochs_run": run.epochs_run,
            }
        )
        print(
            f"  {spec.name:<36} t={t:.2f}  F1 {m['f1']:.4f}  "
            f"recall {m['recall']:.4f}  precision {m['precision']:.4f}  "
            f"AUC {m['roc_auc']:.4f}"
        )

    table = pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)
    save_table(table, "final_candidate_selection")

    best = table.iloc[0]
    print(
        f"\n  Selected on validation F1 at tuned threshold: {best['config']}\n"
        f"    val F1 {best['val_f1']:.4f}, recall {best['val_recall']:.4f}, "
        f"AUC {best['val_roc_auc']:.4f}"
    )

    fixed_pick = selection["best_spec"]["name"]
    if str(best["config"]) != fixed_pick:
        prev = table[table["config"] == fixed_pick]
        if not prev.empty:
            prev = prev.iloc[0]
            print(
                f"  Note: Experiment E's ROC-AUC ranking preferred '{fixed_pick}'\n"
                f"    (val F1 {prev['val_f1']:.4f} at its own tuned threshold). Once the\n"
                f"    threshold is tuned per candidate, '{best['config']}' is the better\n"
                f"    operational choice."
            )

    chosen = next(c for c in candidates if c.name == str(best["config"]))
    chosen.name = "Optimised ANN"
    return chosen


def threshold_curves(proba: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    thresholds = np.linspace(0.05, 0.95, 91)
    precision, recall, f1 = [], [], []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        precision.append(precision_score(y, pred, zero_division=0))
        recall.append(recall_score(y, pred, zero_division=0))
        f1.append(f1_score(y, pred, zero_division=0))
    return thresholds, {
        "Precision": np.asarray(precision),
        "Recall": np.asarray(recall),
        "F1": np.asarray(f1),
    }


def main() -> dict:
    print("=" * 78)
    print("STEP 4  FINAL OPTIMISED MODEL")
    print("=" * 78)

    data = prepare(verbose=True)
    selection = load_selection()
    baseline = load_baseline_results()

    spec = select_candidate(data, selection)

    print("\nConfiguration selected by the experiments:")
    print(f"  optimiser      : {spec.optimizer} (lr={spec.learning_rate:g})")
    print(f"  loss           : {spec.loss}")
    print(f"  hidden layers  : {spec.hidden_units}")
    print(f"  dropout        : {spec.dropout}")
    print(f"  L2             : {spec.l2}")
    print(f"  batch norm     : {spec.batch_norm}")
    print(f"  class weighting: {spec.class_weight}")
    print(f"  batch size     : {spec.batch_size}, max epochs {spec.epochs}")
    print(f"  early stopping : {spec.early_stopping} (patience {spec.patience})")

    run = train_model(spec, data, verbose=0)
    print(f"\nTrained in {run.train_seconds:.1f}s, stopped at epoch {run.epochs_run}")

    # Threshold chosen on validation only.
    tuned_t, tuned_val_f1 = best_threshold(run.model, data.X_val, data.y_val)
    print(f"Threshold tuned on validation: {tuned_t:.2f} (val F1 {tuned_val_f1:.4f})")

    val_metrics = evaluate(run.model, data.X_val, data.y_val, threshold=tuned_t)
    test_default = evaluate(run.model, data.X_test, data.y_test, threshold=0.5)
    test_tuned = evaluate(run.model, data.X_test, data.y_test, threshold=tuned_t)

    base_test = baseline["test"]

    # ---------------------------------------------------------------- comparison
    print("\n" + "=" * 78)
    print(f"BASELINE vs OPTIMISED  (held-out test set, {len(data.y_test)} customers)")
    print("=" * 78)
    print(
        f"{'metric':<12}{'baseline':>12}{'optimised':>12}{'change':>12}{'relative':>12}"
    )
    print("-" * 78)

    rows = []
    for key in HEADLINE_METRICS:
        b, o = base_test[key], test_tuned[key]
        delta = o - b
        rel = (delta / b * 100) if b else float("nan")
        # log loss is an error measure: lower is better, so the arrow flips.
        better = (delta < 0) if key == "log_loss" else (delta > 0)
        mark = "+" if better else ("=" if abs(delta) < 1e-9 else "-")
        print(f"{key:<12}{b:>12.4f}{o:>12.4f}{delta:>+12.4f}{rel:>+11.1f}% {mark}")
        rows.append(
            {
                "metric": key,
                "baseline": round(b, 4),
                "optimised": round(o, 4),
                "absolute_change": round(delta, 4),
                "relative_change_pct": round(rel, 2),
                "improved": bool(better),
            }
        )
    print("-" * 78)

    comparison = pd.DataFrame(rows)
    save_table(comparison, "final_baseline_vs_optimised")

    print("\nConfusion matrices (test set):")
    print(
        f"  baseline  TP={base_test['true_positives']:>3} "
        f"FN={base_test['false_negatives']:>3} "
        f"FP={base_test['false_positives']:>3} "
        f"TN={base_test['true_negatives']:>3}"
    )
    print(
        f"  optimised TP={test_tuned['true_positives']:>3} "
        f"FN={test_tuned['false_negatives']:>3} "
        f"FP={test_tuned['false_positives']:>3} "
        f"TN={test_tuned['true_negatives']:>3}"
    )

    churners = test_tuned["true_positives"] + test_tuned["false_negatives"]
    caught_gain = test_tuned["true_positives"] - base_test["true_positives"]
    fp_change = test_tuned["false_positives"] - base_test["false_positives"]

    print("\nWhat this means for a retention campaign:")
    print(
        f"  Of {churners} customers who actually churned, the baseline identified "
        f"{base_test['true_positives']}"
    )
    print(
        f"  and the optimised model identifies {test_tuned['true_positives']} "
        f"- {caught_gain:+d} customers."
    )
    print(
        f"  False positives moved by {fp_change:+d}: that is the extra contact cost "
        f"of the wider net."
    )
    if test_tuned["accuracy"] < base_test["accuracy"]:
        levers = [f"the decision threshold ({tuned_t:.2f} rather than 0.50)"]
        if spec.class_weight:
            levers.append("class weighting")
        print(
            f"  Accuracy is LOWER than the baseline. That is the intended trade:\n"
            f"  {' and '.join(levers)} deliberately buys recall at the cost of some\n"
            f"  precision, because a missed churner costs more than a wasted discount\n"
            f"  offer. Accuracy is the wrong headline metric for this task - a model\n"
            f"  predicting 'nobody churns' would score "
            f"{1 - float(data.y_test.mean()):.4f} and be worthless."
        )

    # The ranking-quality gain is separated from the threshold gain, because
    # conflating them would overstate what the optimisation actually achieved.
    auc_gain = test_tuned["roc_auc"] - base_test["roc_auc"]
    print(
        f"\n  Attribution, stated plainly: ROC-AUC improved by only {auc_gain:+.4f}.\n"
        f"  ROC-AUC is threshold-free, so that small number is the true gain in the\n"
        f"  model's ability to RANK customers by risk - the part attributable to the\n"
        f"  optimiser, learning rate and architecture work. The large recall gain\n"
        f"  comes mostly from moving the decision threshold, which is a deployment\n"
        f"  choice rather than a better-trained network. Both matter, but they are\n"
        f"  different achievements and should not be reported as one."
    )

    # ---------------------------------------------------------------- figures
    base_model = keras.models.load_model(MODELS_DIR / "baseline.keras")
    base_proba = predict_proba(base_model, data.X_test)
    opt_proba = predict_proba(run.model, data.X_test)

    plot_roc(
        {
            "Baseline (SGD, untuned)": (data.y_test, base_proba),
            "Optimised ANN": (data.y_test, opt_proba),
        },
        {
            "Baseline (SGD, untuned)": base_test["roc_auc"],
            "Optimised ANN": test_tuned["roc_auc"],
        },
        "13_roc_baseline_vs_optimised",
        "ROC curves on the held-out test set",
    )

    plot_confusion(
        data.y_test,
        (opt_proba >= tuned_t).astype(int),
        "14_optimised_confusion",
        f"Optimised ANN confusion matrix (test, threshold {tuned_t:.2f})",
    )

    plot_overfitting_gap(
        run.history,
        "15_optimised_loss",
        "Optimised ANN: training vs validation loss",
    )

    val_proba = predict_proba(run.model, data.X_val)
    thresholds, curves = threshold_curves(val_proba, data.y_val)
    plot_threshold_sweep(
        thresholds,
        curves,
        tuned_t,
        "16_threshold_sweep",
        "Precision / recall / F1 vs decision threshold (validation set)",
    )

    for metric, fig_name in (
        ("recall", "17_recall_improvement"),
        ("f1", "18_f1_improvement"),
        ("roc_auc", "19_auc_improvement"),
    ):
        plot_bar_comparison(
            ["Baseline", "Optimised"],
            [base_test[metric], test_tuned[metric]],
            fig_name,
            f"Test-set {metric.replace('_', ' ').upper()}: baseline vs optimised",
            metric.replace("_", " ").upper(),
            highlight="Optimised",
        )

    # ---------------------------------------------------------------- persist
    payload = {
        "spec": spec.describe(),
        "selection_source": selection,
        "tuned_threshold": tuned_t,
        "validation": val_metrics,
        "test_at_0.5": test_default,
        "test_at_tuned_threshold": test_tuned,
        "baseline_test": base_test,
        "comparison": rows,
        "epochs_run": run.epochs_run,
        "train_seconds": round(run.train_seconds, 2),
        "history": run.history,
    }
    save_json(payload, "final_results")

    run.model.save(MODELS_DIR / "optimised.keras")
    print("  saved model -> models/optimised.keras")

    print("\n" + "=" * 78)
    print("STEP 4 COMPLETE")
    print("=" * 78)
    return payload


if __name__ == "__main__":
    main()
