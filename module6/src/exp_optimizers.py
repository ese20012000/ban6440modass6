"""
Step 2 - optimisation algorithm comparison.

Experiment A holds everything constant except the optimiser, so any difference in
outcome is attributable to the update rule alone. The algorithms are the ones in
the module notes:

  SGD        w <- w - lr * g                      one global step size
  Momentum   v <- b*v + g ; w <- w - lr*v         velocity damps oscillation
  Nesterov   momentum with a look-ahead gradient
  AdaGrad    divides the step by sqrt of accumulated squared gradients
  RMSProp    same idea but a decaying moving average, so the step stops shrinking
  Adam       RMSProp + momentum, with bias correction
  Nadam      Adam with the Nesterov look-ahead

Experiment B varies the batch size to separate the three variants the notes name:
full-batch gradient descent, stochastic (one sample), and mini-batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_prep import prepare
from modeling import ModelSpec, evaluate, results_frame, save_json, save_table, train_model
from plots import plot_bar_comparison, plot_learning_curves

# Held constant across Experiment A so the optimiser is the only variable.
FIXED = dict(
    hidden_units=(16,),
    activation="relu",
    learning_rate=0.01,
    batch_size=32,
    epochs=100,
    loss="binary_crossentropy",
)

OPTIMIZERS = [
    ("SGD", "sgd", {}),
    ("Momentum (0.9)", "momentum", {"momentum": 0.9}),
    ("Nesterov (0.9)", "nesterov", {"momentum": 0.9}),
    ("AdaGrad", "adagrad", {}),
    ("RMSProp", "rmsprop", {}),
    ("Adam", "adam", {}),
    ("Nadam", "nadam", {}),
]

# Set close to the best score any optimiser achieves. A lower bar (0.83) is
# cleared by almost everything in a single epoch and so discriminates nothing.
AUC_TARGET = 0.845


def epochs_to_target(history: dict, target: float = AUC_TARGET) -> int | None:
    """
    First epoch whose validation AUC reaches `target`.

    Final scores hide convergence speed: two optimisers can land in the same place
    with one taking three times as many epochs. In production that difference is
    training cost, so it is worth measuring directly.
    """
    for i, value in enumerate(history.get("val_auc", [])):
        if value >= target:
            return i + 1
    return None


def experiment_a(data) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT A  OPTIMISATION ALGORITHM COMPARISON")
    print("=" * 78)
    print(
        f"Constant: {data.n_features}->16->1, lr={FIXED['learning_rate']}, "
        f"batch={FIXED['batch_size']}, {FIXED['epochs']} epochs, no regularisation"
    )
    print("Variable: the optimiser only\n")

    runs = []
    histories = {}

    for label, key, extra in OPTIMIZERS:
        spec = ModelSpec(name=label, optimizer=key, **FIXED, **extra)
        run = train_model(spec, data, verbose=0)
        run.metrics = evaluate(run.model, data.X_val, data.y_val, threshold=0.5)
        run.metrics["epochs_to_auc_0.83"] = epochs_to_target(run.history) or -1
        run.metrics["final_train_loss"] = run.history["loss"][-1]
        run.metrics["final_val_loss"] = run.history["val_loss"][-1]
        run.metrics["best_val_auc"] = max(run.history["val_auc"])
        runs.append(run)
        histories[label] = run.history

        reached = run.metrics["epochs_to_auc_0.83"]
        print(
            f"  {label:<16} val AUC {run.metrics['roc_auc']:.4f}  "
            f"F1 {run.metrics['f1']:.4f}  "
            f"val loss {run.metrics['final_val_loss']:.4f}  "
            f"epochs->0.83 AUC: {reached if reached > 0 else 'never':<6} "
            f"({run.train_seconds:.0f}s)"
        )

    table = results_frame(runs, "validation")
    save_table(table, "expA_optimizer_comparison")

    plot_learning_curves(
        histories,
        "03_optimizer_loss_curves",
        "Experiment A: binary cross-entropy by optimiser (identical model otherwise)",
        metric="loss",
        ylabel="Binary cross-entropy",
    )
    plot_learning_curves(
        histories,
        "04_optimizer_auc_curves",
        "Experiment A: ROC-AUC by optimiser",
        metric="auc",
        ylabel="ROC-AUC",
    )

    ranked = table.sort_values("roc_auc", ascending=False)
    plot_bar_comparison(
        ranked["model"].tolist(),
        ranked["roc_auc"].tolist(),
        "05_optimizer_auc_bars",
        f"Experiment A: validation ROC-AUC at a shared lr={FIXED['learning_rate']} "
        f"(not a fair ranking - see Experiment C)",
        "Validation ROC-AUC",
        highlight=ranked.iloc[0]["model"],
    )

    winner = ranked.iloc[0]
    print(f"\n  Best at this shared learning rate: {winner['model']} ({winner['roc_auc']:.4f})")
    print(
        "\n  CAVEAT - this comparison is deliberately controlled but NOT fair.\n"
        "  Every optimiser is forced to use lr=0.01. That rate suits plain SGD, but\n"
        "  Adam and RMSProp are normally run near 0.001, so here they take steps that\n"
        "  are roughly ten times too large and are penalised for it. Ranking\n"
        "  algorithms this way measures 'which optimiser happens to like lr=0.01',\n"
        "  not which is better. Experiment C re-runs the comparison with each\n"
        "  optimiser at its own tuned learning rate, which is the fair test."
    )

    fastest = table[table["epochs_to_auc_0.83"] > 0].sort_values("epochs_to_auc_0.83")
    if not fastest.empty:
        f = fastest.iloc[0]
        print(
            f"  Fastest to {AUC_TARGET} validation AUC: {f['model']} "
            f"in {int(f['epochs_to_auc_0.83'])} epochs"
        )
    never = table[table["epochs_to_auc_0.83"] < 0]["model"].tolist()
    if never:
        print(f"  Never reached {AUC_TARGET} AUC in {FIXED['epochs']} epochs: {', '.join(never)}")

    save_json(
        {
            "fixed_settings": {k: list(v) if isinstance(v, tuple) else v for k, v in FIXED.items()},
            "histories": histories,
            "auc_target": AUC_TARGET,
        },
        "expA_histories",
    )
    return table


# ---------------------------------------------------------------------------
# Experiment B - batch size / gradient descent variant
# ---------------------------------------------------------------------------
def experiment_b(data) -> pd.DataFrame:
    print("\n" + "=" * 78)
    print("EXPERIMENT B  BATCH SIZE: FULL-BATCH vs MINI-BATCH vs STOCHASTIC")
    print("=" * 78)
    print("Plain SGD throughout, so the batching strategy is the only variable.")
    print("Note: at equal epochs, larger batches make proportionally fewer weight")
    print("updates - which is the mechanism the notes describe, shown directly.\n")

    n_train = len(data.X_train)
    configs = [
        ("Mini-batch 16", 16, 60),
        ("Mini-batch 32", 32, 60),
        ("Mini-batch 64", 64, 60),
        ("Mini-batch 128", 128, 60),
        ("Full-batch GD", n_train, 60),
    ]

    runs = []
    histories = {}

    for label, batch, epochs in configs:
        spec = ModelSpec(
            name=label,
            hidden_units=(16,),
            optimizer="sgd",
            learning_rate=0.01,
            batch_size=batch,
            epochs=epochs,
        )
        run = train_model(spec, data, verbose=0)
        run.metrics = evaluate(run.model, data.X_val, data.y_val, threshold=0.5)
        run.metrics["batch_size"] = batch
        run.metrics["updates_per_epoch"] = int(np.ceil(n_train / batch))
        run.metrics["total_updates"] = run.metrics["updates_per_epoch"] * run.epochs_run
        run.metrics["seconds_per_epoch"] = round(run.train_seconds / max(run.epochs_run, 1), 3)
        runs.append(run)
        histories[label] = run.history

        print(
            f"  {label:<16} val AUC {run.metrics['roc_auc']:.4f}  "
            f"updates/epoch {run.metrics['updates_per_epoch']:>4}  "
            f"total updates {run.metrics['total_updates']:>6}  "
            f"{run.metrics['seconds_per_epoch']:.2f}s/epoch"
        )

    # True stochastic gradient descent (batch size 1) is run briefly and reported
    # separately. At 4507 updates per epoch it is far too slow to train to
    # convergence here, which is itself the finding worth recording.
    print("\n  Stochastic GD (batch=1), 3 epochs only - measuring cost per update:")
    sgd_spec = ModelSpec(
        name="Stochastic GD (batch=1)",
        hidden_units=(16,),
        optimizer="sgd",
        learning_rate=0.01,
        batch_size=1,
        epochs=3,
    )
    sgd_run = train_model(sgd_spec, data, verbose=0)
    sgd_run.metrics = evaluate(sgd_run.model, data.X_val, data.y_val, threshold=0.5)
    sgd_run.metrics["batch_size"] = 1
    sgd_run.metrics["updates_per_epoch"] = n_train
    sgd_run.metrics["total_updates"] = n_train * sgd_run.epochs_run
    sgd_run.metrics["seconds_per_epoch"] = round(
        sgd_run.train_seconds / max(sgd_run.epochs_run, 1), 3
    )
    print(
        f"  {'Stochastic (1)':<16} val AUC {sgd_run.metrics['roc_auc']:.4f}  "
        f"updates/epoch {n_train:>4}  "
        f"{sgd_run.metrics['seconds_per_epoch']:.2f}s/epoch  "
        f"({sgd_run.metrics['seconds_per_epoch'] / max(runs[1].metrics['seconds_per_epoch'], 1e-9):.0f}x "
        f"the cost of batch 32 per epoch)"
    )
    runs.append(sgd_run)
    histories["Stochastic (batch=1), 3 epochs"] = sgd_run.history

    table = results_frame(runs, "validation")
    save_table(table, "expB_batch_size")

    plot_learning_curves(
        histories,
        "06_batch_size_loss_curves",
        "Experiment B: effect of batch size on convergence (plain SGD)",
        metric="loss",
        ylabel="Binary cross-entropy",
    )

    full = table[table["model"] == "Full-batch GD"].iloc[0]
    mini = table[table["model"] == "Mini-batch 32"].iloc[0]
    print(
        f"\n  At 60 epochs, full-batch GD made {int(full['total_updates'])} updates "
        f"vs {int(mini['total_updates'])} for mini-batch 32,"
    )
    print(
        f"  giving validation AUC {full['roc_auc']:.4f} vs {mini['roc_auc']:.4f}. "
        f"Same gradient, far fewer steps."
    )

    return table


def main() -> None:
    data = prepare(verbose=True)
    table_a = experiment_a(data)
    table_b = experiment_b(data)

    print("\n" + "=" * 78)
    print("STEP 2 COMPLETE")
    print("=" * 78)
    best = table_a.sort_values("roc_auc", ascending=False).iloc[0]
    print(
        f"At a shared lr=0.01 the leader is {best['model']} ({best['roc_auc']:.4f}), "
        f"but that rate\nfavours it by construction."
    )
    print(
        "Next: Experiment C sweeps the learning rate for every optimiser "
        "independently\nso each is judged at its own best setting."
    )


if __name__ == "__main__":
    main()
