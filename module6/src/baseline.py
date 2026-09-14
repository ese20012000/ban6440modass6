"""
Step 1 - the baseline artificial neural network.

The assignment brief refers to an "existing" ANN but supplies none, so this
constructs the baseline explicitly (as directed by the course instructor). It is
deliberately naive - the kind of first-pass model someone writes before thinking
about optimisation:

  * one hidden layer of 16 ReLU units
  * plain mini-batch gradient descent (SGD, momentum = 0), learning rate 0.01
  * no regularisation, no early stopping, no class weighting
  * a fixed 100 epochs and the default 0.5 decision threshold

Every one of those choices is revisited in the experiments that follow. Nothing
here is tuned, which is the point: it establishes the reference that later gains
are measured against.
"""

from __future__ import annotations

from config import MODELS_DIR
from data_prep import prepare
from modeling import (
    ModelSpec,
    best_threshold,
    evaluate,
    save_json,
    train_model,
)
from plots import plot_confusion, plot_overfitting_gap

BASELINE_SPEC = ModelSpec(
    name="Baseline (SGD, lr=0.01)",
    hidden_units=(16,),
    activation="relu",
    dropout=0.0,
    l2=0.0,
    batch_norm=False,
    optimizer="sgd",
    learning_rate=0.01,
    loss="binary_crossentropy",
    batch_size=32,
    epochs=100,
    early_stopping=False,
    class_weight=False,
    notes="Untuned reference model. Plain gradient descent, no regularisation.",
)


def main() -> dict:
    print("=" * 78)
    print("STEP 1  BASELINE ARTIFICIAL NEURAL NETWORK")
    print("=" * 78)

    data = prepare(verbose=True)

    print(f"\nTraining: {BASELINE_SPEC.name}")
    print(
        f"  architecture : {data.n_features} -> "
        f"{' -> '.join(str(u) for u in BASELINE_SPEC.hidden_units)} -> 1 (sigmoid)"
    )
    print(f"  optimiser    : plain SGD, lr={BASELINE_SPEC.learning_rate}, momentum=0")
    print(f"  loss         : {BASELINE_SPEC.loss}")
    print(f"  epochs       : {BASELINE_SPEC.epochs}, batch size {BASELINE_SPEC.batch_size}")

    run = train_model(BASELINE_SPEC, data, verbose=0)
    run.model.summary()

    val_metrics = evaluate(run.model, data.X_val, data.y_val, threshold=0.5)
    test_metrics = evaluate(run.model, data.X_test, data.y_test, threshold=0.5)

    # Reported for context only - the baseline is scored at 0.5, as an untuned
    # model would be. It shows how much headroom the threshold alone represents.
    tuned_t, tuned_f1 = best_threshold(run.model, data.X_val, data.y_val)

    print(f"\nTrained in {run.train_seconds:.1f}s over {run.epochs_run} epochs")
    print("\nBaseline performance @ threshold 0.50")
    print("-" * 78)
    print(f"{'metric':<14}{'validation':>14}{'test':>14}")
    print("-" * 78)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss"):
        print(f"{key:<14}{val_metrics[key]:>14.4f}{test_metrics[key]:>14.4f}")
    print("-" * 78)
    print(
        f"Test confusion: TP={test_metrics['true_positives']} "
        f"FN={test_metrics['false_negatives']} "
        f"FP={test_metrics['false_positives']} "
        f"TN={test_metrics['true_negatives']}"
    )

    majority = 1.0 - float(data.y_test.mean())
    print(
        f"\nSanity check: predicting 'no churn' for everyone scores {majority:.4f} "
        f"accuracy.\n  The baseline's {test_metrics['accuracy']:.4f} must be read "
        f"against that, not against zero."
    )
    print(
        f"  Recall of {test_metrics['recall']:.4f} means it misses "
        f"{test_metrics['false_negatives']} of "
        f"{test_metrics['false_negatives'] + test_metrics['true_positives']} "
        f"real churners - the customers a retention campaign exists to reach."
    )
    print(f"  F1 at a tuned threshold ({tuned_t:.2f}) would be {tuned_f1:.4f} on validation.")

    plot_overfitting_gap(
        run.history,
        "01_baseline_loss",
        "Baseline: training vs validation loss (plain SGD)",
    )
    plot_confusion(
        data.y_test,
        (run.model.predict(data.X_test, verbose=0).ravel() >= 0.5).astype(int),
        "02_baseline_confusion",
        "Baseline confusion matrix (test set, threshold 0.50)",
    )

    payload = {
        "spec": BASELINE_SPEC.describe(),
        "validation": val_metrics,
        "test": test_metrics,
        "tuned_threshold_reference": {"threshold": tuned_t, "val_f1": tuned_f1},
        "epochs_run": run.epochs_run,
        "train_seconds": round(run.train_seconds, 2),
        "history": run.history,
        "majority_class_accuracy": majority,
    }
    save_json(payload, "baseline_results")

    run.model.save(MODELS_DIR / "baseline.keras")
    print(f"  saved model -> models/baseline.keras")

    return payload


if __name__ == "__main__":
    main()
