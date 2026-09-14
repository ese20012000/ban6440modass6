"""
Step 4b - test-set evaluation of the optimised architecture under every optimiser.

The brief requires the optimised model to be assessed for at least three
optimisation algorithms. Experiments A and C did that on validation data; this
script reports it on the held-out test set, which is what the submission asks for.

Method: the architecture is frozen at the configuration selected in Step 4, and the
only thing that changes is the optimiser - each one paired with the learning rate
Experiment C found best for it. Every model gets its own threshold tuned on
validation, then is scored once on test.

One caveat stated up front. Scoring seven models on the test set invites
multiple-comparison bias: the best of seven test scores is optimistically biased
even if all seven are equivalent. The primary model was therefore committed to on
validation evidence alone, before any of these test numbers existed. This table is
a comparative report, not a second selection round, and the pre-registered choice
is flagged in the output.
"""

from __future__ import annotations

import json

import pandas as pd

from config import TABLES_DIR
from data_prep import prepare
from modeling import (
    ModelSpec,
    best_threshold,
    evaluate,
    save_json,
    save_table,
    train_model,
)
from plots import plot_bar_comparison

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss"]


def load_json(name: str) -> dict:
    path = TABLES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{name}.json not found - run the earlier steps first.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> dict:
    print("=" * 78)
    print("STEP 4b  OPTIMISED ARCHITECTURE EVALUATED UNDER EACH OPTIMISER (TEST SET)")
    print("=" * 78)

    data = prepare(verbose=True)
    final = load_json("final_results")
    baseline = load_json("baseline_results")
    base_test = baseline["test"]

    primary_optimizer = final["spec"]["optimizer"]
    best_lrs = pd.read_csv(TABLES_DIR / "expC_optimizer_best.csv")

    # Freeze the architecture selected in Step 4. Only the optimiser varies.
    frozen = dict(final["spec"])
    frozen["hidden_units"] = tuple(frozen["hidden_units"])

    print(f"\nFrozen architecture: {frozen['hidden_units']} hidden, "
          f"dropout {frozen['dropout']}, L2 {frozen['l2']}, "
          f"batch norm {frozen['batch_norm']}, class weight {frozen['class_weight']}")
    print(f"Loss: {frozen['loss']}, batch size {frozen['batch_size']}")
    print(f"Pre-registered primary optimiser (chosen on validation): {primary_optimizer}\n")

    rows = []
    for _, r in best_lrs.iterrows():
        label, key, lr = str(r["optimizer"]), str(r["optimizer_key"]), float(r["best_learning_rate"])

        spec_dict = dict(frozen)
        spec_dict.update(
            name=f"Optimised ANN + {label}",
            optimizer=key,
            learning_rate=lr,
            momentum=0.9,
        )
        spec = ModelSpec(**spec_dict)

        run = train_model(spec, data, verbose=0)
        t, _ = best_threshold(run.model, data.X_val, data.y_val)
        test = evaluate(run.model, data.X_test, data.y_test, threshold=t)

        row = {
            "optimizer": label,
            "learning_rate": lr,
            "tuned_threshold": round(t, 3),
            "is_primary": key == primary_optimizer,
            "epochs_run": run.epochs_run,
            "train_seconds": round(run.train_seconds, 2),
        }
        row.update({m: round(test[m], 4) for m in METRICS})
        row.update(
            {
                "true_positives": test["true_positives"],
                "false_negatives": test["false_negatives"],
                "false_positives": test["false_positives"],
                "true_negatives": test["true_negatives"],
            }
        )
        rows.append(row)

        flag = "  <- primary" if row["is_primary"] else ""
        print(
            f"  {label:<16} lr={lr:<7g} t={t:.2f}  "
            f"acc {test['accuracy']:.4f}  prec {test['precision']:.4f}  "
            f"rec {test['recall']:.4f}  F1 {test['f1']:.4f}  "
            f"AUC {test['roc_auc']:.4f}  loss {test['log_loss']:.4f}{flag}"
        )

    table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    # Prepend the baseline so the table reads as a single before/after comparison.
    base_row = {
        "optimizer": "BASELINE (plain SGD, untuned)",
        "learning_rate": 0.01,
        "tuned_threshold": 0.5,
        "is_primary": False,
        "epochs_run": baseline["epochs_run"],
        "train_seconds": baseline["train_seconds"],
    }
    base_row.update({m: round(base_test[m], 4) for m in METRICS})
    base_row.update(
        {
            "true_positives": base_test["true_positives"],
            "false_negatives": base_test["false_negatives"],
            "false_positives": base_test["false_positives"],
            "true_negatives": base_test["true_negatives"],
        }
    )
    full = pd.concat([pd.DataFrame([base_row]), table], ignore_index=True)
    save_table(full, "step4_optimizer_test_evaluation")

    # ------------------------------------------------------------------ summary
    print("\n" + "=" * 78)
    print("BEFORE AND AFTER, TEST SET  (all optimisers at their tuned lr and threshold)")
    print("=" * 78)
    header = f"{'model':<32}" + "".join(f"{m[:8]:>10}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for _, r in full.iterrows():
        line = f"{str(r['optimizer'])[:31]:<32}"
        line += "".join(f"{r[m]:>10.4f}" for m in METRICS)
        print(line)
    print("-" * len(header))

    print("\nChurners identified out of 374 in the test set:")
    for _, r in full.iterrows():
        print(
            f"  {str(r['optimizer'])[:34]:<36}{int(r['true_positives']):>4} caught, "
            f"{int(r['false_negatives']):>4} missed, "
            f"{int(r['false_positives']):>4} false alarms"
        )

    improved = table[table["f1"] > base_test["f1"]]
    print(
        f"\n  {len(improved)} of {len(table)} optimisers beat the baseline F1 "
        f"({base_test['f1']:.4f})."
    )
    print(
        f"  Spread in test F1 across the seven: "
        f"{table['f1'].min():.4f} to {table['f1'].max():.4f} "
        f"({table['f1'].max() - table['f1'].min():.4f})."
    )
    print(
        "  Once each optimiser is given a suitable learning rate, the choice of\n"
        "  algorithm matters far less than the fact that it was tuned at all."
    )

    primary = table[table["is_primary"]]
    if not primary.empty:
        p = primary.iloc[0]
        rank = int(table.index[table["is_primary"]][0]) + 1
        print(
            f"\n  The pre-registered primary ({p['optimizer']}) ranks {rank} of "
            f"{len(table)} on test F1.\n"
            f"  It was selected on validation data only, so this ranking is an\n"
            f"  unbiased check rather than a selection - reporting the best of seven\n"
            f"  test scores as 'the result' would be exactly the bias this avoids."
        )

    # ------------------------------------------------------------------ figures
    labels = ["Baseline"] + table["optimizer"].tolist()
    for metric, fig in (("f1", "24_step4_f1_by_optimizer"), ("recall", "25_step4_recall_by_optimizer")):
        plot_bar_comparison(
            labels,
            [base_test[metric]] + table[metric].tolist(),
            fig,
            f"Test-set {metric.upper()}: baseline vs optimised model under each optimiser",
            metric.upper(),
            highlight=primary.iloc[0]["optimizer"] if not primary.empty else None,
        )

    payload = {
        "frozen_architecture": {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in frozen.items()
        },
        "primary_optimizer": primary_optimizer,
        "baseline_test": base_test,
        "results": full.to_dict("records"),
    }
    save_json(payload, "step4_optimizer_test_evaluation")

    print("\n" + "=" * 78)
    print("STEP 4b COMPLETE")
    print("=" * 78)
    return payload


if __name__ == "__main__":
    main()
