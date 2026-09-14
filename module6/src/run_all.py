"""
Run the whole pipeline end to end, in order.

    python src/run_all.py

Each step depends on files written by the one before it, so the order is not
optional. Expect roughly 30-40 minutes on CPU; Experiment C alone trains 42 models.
"""

from __future__ import annotations

import time

import baseline
import exp_final_evaluation
import exp_optimizers
import exp_tuning
import final_model
import segmentation

STEPS = [
    ("1  Baseline ANN", baseline.main),
    ("2  Optimiser and batch-size experiments", exp_optimizers.main),
    ("3  Learning rate, loss function, architecture tuning", exp_tuning.main),
    ("4  Final optimised model and comparison", final_model.main),
    ("4b All optimisers evaluated on the test set", exp_final_evaluation.main),
    ("5  Marketing segmentation", segmentation.main),
]


def main() -> None:
    started = time.perf_counter()

    for label, fn in STEPS:
        print("\n\n")
        print("#" * 78)
        print(f"#  STEP {label}")
        print("#" * 78)
        step_start = time.perf_counter()
        fn()
        print(f"\n[step finished in {time.perf_counter() - step_start:.0f}s]")

    total = time.perf_counter() - started
    print("\n" + "#" * 78)
    print(f"#  PIPELINE COMPLETE in {total / 60:.1f} minutes")
    print("#" * 78)
    print("\nOutputs:")
    print("  results/tables/   metrics, comparisons, segment profiles (CSV + JSON)")
    print("  results/figures/  25 figures")
    print("  models/           baseline.keras, optimised.keras")


if __name__ == "__main__":
    main()
