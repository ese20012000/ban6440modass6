"""
Step 3 - tuning the winning optimiser.

Experiment C  learning rate sweep
Experiment D  loss function: cross-entropy vs MSE vs MAE
Experiment E  architecture and regularisation

Each stage inherits the best configuration found by the previous one. This is a
greedy coordinate search rather than an exhaustive grid: cheaper, and it keeps the
reasoning chain legible, which matters more here than squeezing out a final
fraction of a percent.
"""

from __future__ import annotations

import pandas as pd

from data_prep import prepare
from modeling import ModelSpec, evaluate, results_frame, save_json, save_table, train_model
from plots import plot_bar_comparison, plot_learning_curves

LEARNING_RATES = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

OPTIMIZERS = [
    ("SGD", "sgd", {}),
    ("Momentum (0.9)", "momentum", {"momentum": 0.9}),
    ("Nesterov (0.9)", "nesterov", {"momentum": 0.9}),
    ("AdaGrad", "adagrad", {}),
    ("RMSProp", "rmsprop", {}),
    ("Adam", "adam", {}),
    ("Nadam", "nadam", {}),
]

LOSSES = [
    ("Binary cross-entropy", "binary_crossentropy"),
    ("Mean squared error", "mse"),
    ("Mean absolute error", "mae"),
]


# ---------------------------------------------------------------------------
# Experiment C - fair optimiser comparison, each at its own best learning rate
# ---------------------------------------------------------------------------
def experiment_c(data) -> tuple[str, str, float]:
    """
    Sweep the learning rate independently for every optimiser.

    Experiment A held the learning rate fixed, which is a controlled test but an
    unfair ranking: the shared rate inevitably suits some update rules better than
    others. Here each optimiser is given the same grid of rates and judged at its
    own best, which is the comparison that actually answers "which algorithm is
    better for this problem".

    The cost is 7 x 6 = 42 runs. Early stopping keeps that tractable by cutting off
    configurations that have stopped improving - including the diverging ones.
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT C  FAIR OPTIMISER COMPARISON (learning rate tuned per optimiser)")
    print("=" * 78)
    print(f"Grid: {len(OPTIMIZERS)} optimisers x {len(LEARNING_RATES)} learning rates")
    print("The learning rate sets how far each step moves: too small and training")
    print("crawls, too large and the updates overshoot and the loss oscillates.")
    print("The best rate differs by algorithm, which is exactly why Experiment A's")
    print("single shared rate could not rank them fairly.\n")

    rows = []
    best_per_optimizer: dict[str, dict] = {}
    curves_for_plot: dict[str, dict] = {}

    for label, key, extra in OPTIMIZERS:
        best_here = None

        for lr in LEARNING_RATES:
            spec = ModelSpec(
                name=f"{label} lr={lr:g}",
                hidden_units=(16,),
                optimizer=key,
                learning_rate=lr,
                batch_size=32,
                epochs=100,
                early_stopping=True,
                patience=12,
                **extra,
            )
            run = train_model(spec, data, verbose=0)
            metrics = evaluate(run.model, data.X_val, data.y_val, threshold=0.5)
            best_val_auc = max(run.history["val_auc"])

            rows.append(
                {
                    "optimizer": label,
                    "optimizer_key": key,
                    "learning_rate": lr,
                    "best_val_auc": best_val_auc,
                    "val_auc_final": metrics["roc_auc"],
                    "val_f1": metrics["f1"],
                    "val_recall": metrics["recall"],
                    "min_val_loss": min(run.history["val_loss"]),
                    "epochs_run": run.epochs_run,
                    "train_seconds": round(run.train_seconds, 2),
                }
            )

            if best_here is None or best_val_auc > best_here["best_val_auc"]:
                best_here = {
                    "optimizer": label,
                    "optimizer_key": key,
                    "learning_rate": lr,
                    "best_val_auc": best_val_auc,
                    "val_f1": metrics["f1"],
                    "history": run.history,
                }

        best_per_optimizer[label] = best_here
        curves_for_plot[f"{label} (lr={best_here['learning_rate']:g})"] = best_here["history"]
        print(
            f"  {label:<16} best lr={best_here['learning_rate']:<7g} "
            f"val AUC {best_here['best_val_auc']:.4f}  F1 {best_here['val_f1']:.4f}"
        )

    grid = pd.DataFrame(rows)
    save_table(grid, "expC_optimizer_lr_grid")

    # Wide pivot: optimisers as rows, learning rates as columns. This is the table
    # that shows at a glance how differently each algorithm responds to the rate.
    pivot = grid.pivot_table(
        index="optimizer", columns="learning_rate", values="best_val_auc"
    ).round(4)
    save_table(pivot, "expC_lr_sensitivity_pivot", index=True)
    print("\n  Best validation ROC-AUC by optimiser x learning rate:")
    print(pivot.to_string())

    summary = (
        pd.DataFrame(
            [
                {
                    "optimizer": v["optimizer"],
                    "optimizer_key": v["optimizer_key"],
                    "best_learning_rate": v["learning_rate"],
                    "best_val_auc": v["best_val_auc"],
                    "val_f1": v["val_f1"],
                }
                for v in best_per_optimizer.values()
            ]
        )
        .sort_values("best_val_auc", ascending=False)
        .reset_index(drop=True)
    )
    save_table(summary, "expC_optimizer_best")

    plot_learning_curves(
        curves_for_plot,
        "07_optimizer_tuned_curves",
        "Experiment C: each optimiser at its own tuned learning rate",
        metric="loss",
        ylabel="Binary cross-entropy",
    )
    plot_bar_comparison(
        summary["optimizer"].tolist(),
        summary["best_val_auc"].tolist(),
        "08_optimizer_tuned_bars",
        "Experiment C: validation ROC-AUC with the learning rate tuned per optimiser",
        "Best validation ROC-AUC",
        highlight=summary.iloc[0]["optimizer"],
    )

    winner = summary.iloc[0]
    print(
        f"\n  Fair winner: {winner['optimizer']} at lr={winner['best_learning_rate']:g} "
        f"(val AUC {winner['best_val_auc']:.4f})"
    )
    spread = summary["best_val_auc"].max() - summary["best_val_auc"].min()
    print(
        f"  Spread across optimisers once each is tuned: {spread:.4f} AUC.\n"
        f"  Compare that with the spread in Experiment A, where the shared rate\n"
        f"  created a much larger and largely artificial gap."
    )

    return str(winner["optimizer"]), str(winner["optimizer_key"]), float(
        winner["best_learning_rate"]
    )


# ---------------------------------------------------------------------------
# Experiment D - loss function
# ---------------------------------------------------------------------------
def experiment_d(data, opt_label: str, opt_key: str, lr: float) -> str:
    print("\n" + "=" * 78)
    print("EXPERIMENT D  LOSS FUNCTION: CROSS-ENTROPY vs MSE vs MAE")
    print("=" * 78)
    print("This is a classification problem, so cross-entropy is the principled")
    print("choice. MSE is included because it is the regression loss from the module")
    print("notes, and running it shows *why* the distinction matters rather than")
    print("just asserting it.\n")

    runs, histories = [], {}

    for label, loss in LOSSES:
        spec = ModelSpec(
            name=label,
            hidden_units=(16,),
            optimizer=opt_key,
            momentum=0.9,
            learning_rate=lr,
            batch_size=32,
            epochs=80,
            loss=loss,
        )
        run = train_model(spec, data, verbose=0)
        # Scored on ROC-AUC and F1, which are loss-agnostic. Comparing the raw loss
        # values across different loss functions would be meaningless - they are
        # measured on different scales.
        run.metrics = evaluate(run.model, data.X_val, data.y_val, threshold=0.5)
        run.metrics["loss_function"] = loss
        run.metrics["best_val_auc"] = max(run.history["val_auc"])
        runs.append(run)
        histories[label] = run.history

        print(
            f"  {label:<22} val AUC {run.metrics['roc_auc']:.4f}  "
            f"F1 {run.metrics['f1']:.4f}  recall {run.metrics['recall']:.4f}  "
            f"cross-entropy {run.metrics['log_loss']:.4f}"
        )

    table = results_frame(runs, "validation")
    save_table(table, "expD_loss_function")

    plot_learning_curves(
        histories,
        "09_loss_function_auc_curves",
        "Experiment D: validation ROC-AUC by loss function",
        metric="auc",
        ylabel="ROC-AUC",
    )

    ranked = table.sort_values("best_val_auc", ascending=False)
    plot_bar_comparison(
        ranked["model"].tolist(),
        ranked["best_val_auc"].tolist(),
        "10_loss_function_bars",
        "Experiment D: best validation ROC-AUC by loss function",
        "Best validation ROC-AUC",
        highlight=ranked.iloc[0]["model"],
    )

    best_row = ranked.iloc[0]
    best_loss = str(best_row["loss_function"])
    print(f"\n  Best: {best_row['model']} (val AUC {best_row['best_val_auc']:.4f})")
    print(
        "  Note the comparable cross-entropy column: it is reported for every run\n"
        "  regardless of what was minimised, which is what makes them comparable."
    )
    return best_loss


# ---------------------------------------------------------------------------
# Experiment E - architecture and regularisation
# ---------------------------------------------------------------------------
def architecture_configs(opt_key: str, lr: float, loss: str) -> list[ModelSpec]:
    """
    The Experiment E candidate list.

    Exposed as a function rather than buried inside the experiment so that
    final_model.py can rebuild exactly these candidates for its own selection step
    instead of duplicating the definitions.
    """
    base = dict(
        optimizer=opt_key,
        momentum=0.9,
        learning_rate=lr,
        loss=loss,
        batch_size=32,
        epochs=150,
        early_stopping=True,
        patience=15,
    )

    return [
        ModelSpec(name="16 (carried forward)", hidden_units=(16,), **base),
        ModelSpec(name="32-16", hidden_units=(32, 16), **base),
        ModelSpec(name="64-32", hidden_units=(64, 32), **base),
        ModelSpec(name="64-32 + dropout 0.3", hidden_units=(64, 32), dropout=0.3, **base),
        ModelSpec(name="64-32 + L2 1e-4", hidden_units=(64, 32), l2=1e-4, **base),
        ModelSpec(
            name="64-32 + dropout + batchnorm",
            hidden_units=(64, 32),
            dropout=0.3,
            batch_norm=True,
            **base,
        ),
        ModelSpec(
            name="32-16 + dropout 0.2 + class weight",
            hidden_units=(32, 16),
            dropout=0.2,
            class_weight=True,
            **base,
        ),
    ]


def experiment_e(
    data, opt_label: str, opt_key: str, lr: float, loss: str
) -> tuple[pd.DataFrame, ModelSpec]:
    print("\n" + "=" * 78)
    print("EXPERIMENT E  ARCHITECTURE AND REGULARISATION")
    print("=" * 78)
    print("Early stopping on validation AUC is enabled for all runs here, so each")
    print("configuration is judged at its own best epoch rather than at an arbitrary")
    print("fixed one. Capacity increases down the list; regularisation is then added")
    print("to control the overfitting that extra capacity invites.")
    print()
    print("Caveat carried into Step 4: the F1 and recall columns below are measured at")
    print("a fixed 0.5 threshold. Class weighting changes the calibration of the")
    print("output probabilities, so it looks dramatically better on recall partly for")
    print("that reason alone. final_model.py re-compares these candidates at each")
    print("one's own tuned threshold before committing to a winner.\n")

    configs = architecture_configs(opt_key, lr, loss)

    runs, histories = [], {}
    by_name = {spec.name: spec for spec in configs}

    for spec in configs:
        run = train_model(spec, data, verbose=0)
        run.metrics = evaluate(run.model, data.X_val, data.y_val, threshold=0.5)
        run.metrics["best_val_auc"] = max(run.history["val_auc"])
        run.metrics["params"] = int(run.model.count_params())
        runs.append(run)
        histories[spec.name] = run.history

        print(
            f"  {spec.name:<36} val AUC {run.metrics['roc_auc']:.4f}  "
            f"F1 {run.metrics['f1']:.4f}  recall {run.metrics['recall']:.4f}  "
            f"stopped @ {run.epochs_run:>3} epochs  "
            f"{run.metrics['params']:>5} params"
        )

    table = results_frame(runs, "validation")
    save_table(table, "expE_architecture_regularisation")

    plot_learning_curves(
        histories,
        "11_architecture_loss_curves",
        "Experiment E: architecture and regularisation",
        metric="loss",
        ylabel="Loss",
    )

    ranked = table.sort_values("best_val_auc", ascending=False)
    plot_bar_comparison(
        ranked["model"].tolist(),
        ranked["best_val_auc"].tolist(),
        "12_architecture_bars",
        "Experiment E: best validation ROC-AUC by configuration",
        "Best validation ROC-AUC",
        highlight=ranked.iloc[0]["model"],
    )

    best = ranked.iloc[0]
    best_spec = by_name[str(best["model"])]
    print(f"\n  Best configuration: {best['model']} (val AUC {best['best_val_auc']:.4f})")

    save_json(
        {
            "optimizer": opt_key,
            "optimizer_label": opt_label,
            "learning_rate": lr,
            "loss": loss,
            "best_config": str(best["model"]),
            "histories": histories,
        },
        "expE_histories",
    )
    return table, best_spec


def main() -> None:
    data = prepare(verbose=True)

    opt_label, opt_key, best_lr = experiment_c(data)
    best_loss = experiment_d(data, opt_label, opt_key, best_lr)
    _, best_spec = experiment_e(data, opt_label, opt_key, best_lr, best_loss)

    # The full winning spec is written out so final_model.py rebuilds exactly this
    # configuration rather than a hand-copied approximation of it.
    save_json(
        {
            "optimizer": opt_key,
            "optimizer_label": opt_label,
            "learning_rate": best_lr,
            "loss": best_loss,
            "best_spec": best_spec.describe(),
        },
        "tuning_selection",
    )

    print("\n" + "=" * 78)
    print("STEP 3 COMPLETE")
    print("=" * 78)
    print(f"Selected: optimiser={opt_key}, lr={best_lr:g}, loss={best_loss}")
    print(f"Architecture: {best_spec.name}")
    print("Next: train the final model and evaluate it on the held-out test set.")


if __name__ == "__main__":
    main()
