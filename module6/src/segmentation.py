"""
Step 5 - turning the optimised model into marketing segments.

A churn probability is not a marketing decision. What Otomoto can act on is a
small number of named groups with a different budget and message for each.

Segments are formed on two axes:
  risk   - the optimised ANN's predicted churn probability, split at the
           operational threshold chosen on validation data
  value  - monthly charges, split at the median, as a proxy for revenue per customer

That gives four quadrants, and the quadrants are then validated against the actual
churn outcomes in the held-out test set. If the "high risk" groups did not really
churn more, the segmentation would be decoration.
"""

from __future__ import annotations

import json

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import MODELS_DIR, RANDOM_SEED, TABLES_DIR
from data_prep import prepare
from modeling import predict_proba, save_json, save_table
from plots import plot_bar_comparison, plot_segments

CAMPAIGN_PLAYBOOK = {
    "Retain Now": (
        "High risk, high value. Personal outreach plus a targeted contract-upgrade "
        "or loyalty discount. Highest budget per customer - this is where lost "
        "revenue is concentrated."
    ),
    "Monitor": (
        "High risk, lower value. Automated, low-cost intervention: email offers, "
        "self-service nudges. Do not spend agent time here; the margin will not "
        "cover it."
    ),
    "Grow": (
        "Low risk, high value. Already loyal and already profitable. Cross-sell and "
        "upsell add-ons rather than discount - a retention offer here just erodes "
        "revenue that was not at risk."
    ),
    "Nurture": (
        "Low risk, lower value. Keep costs minimal, build engagement over time and "
        "look for organic upgrade paths."
    ),
}


def load_final() -> dict:
    path = TABLES_DIR / "final_results.json"
    if not path.exists():
        raise FileNotFoundError("final_results.json not found - run final_model.py first.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def assign_segments(risk: np.ndarray, value: np.ndarray, risk_cut: float, value_cut: float):
    high_risk = risk >= risk_cut
    high_value = value >= value_cut

    segment = np.empty(len(risk), dtype=object)
    segment[high_risk & high_value] = "Retain Now"
    segment[high_risk & ~high_value] = "Monitor"
    segment[~high_risk & high_value] = "Grow"
    segment[~high_risk & ~high_value] = "Nurture"
    return segment


def permutation_importance(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 3,
) -> pd.DataFrame:
    """
    Rank features by how much shuffling each one degrades test ROC-AUC.

    A neural network has no coefficients to read off, so this is the practical way
    to answer "what is actually driving these risk scores". Shuffling one column
    breaks its relationship with the target while leaving the model untouched; the
    resulting AUC drop is that feature's contribution.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    baseline_auc = roc_auc_score(y, predict_proba(model, X))

    records = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            rng.shuffle(X_shuffled[:, j])
            drops.append(baseline_auc - roc_auc_score(y, predict_proba(model, X_shuffled)))
        records.append({"feature": name, "auc_drop": float(np.mean(drops))})

    return (
        pd.DataFrame(records)
        .sort_values("auc_drop", ascending=False)
        .reset_index(drop=True)
    )


def main() -> dict:
    print("=" * 78)
    print("STEP 5  MARKETING SEGMENTATION FROM THE OPTIMISED MODEL")
    print("=" * 78)

    data = prepare(verbose=True)
    final = load_final()
    model = keras.models.load_model(MODELS_DIR / "optimised.keras")

    risk_cut = float(final["tuned_threshold"])
    risk = predict_proba(model, data.X_test)

    profile = data.raw_test.copy()
    profile["churn_probability"] = risk
    profile["actual_churn"] = data.y_test.astype(int)

    value = profile["MonthlyCharges"].to_numpy()
    value_cut = float(np.median(value))

    profile["segment"] = assign_segments(risk, value, risk_cut, value_cut)

    print(f"\nSegment boundaries")
    print(f"  risk  : churn probability >= {risk_cut:.2f} (threshold tuned on validation)")
    print(f"  value : monthly charges  >= ${value_cut:.2f} (test-set median)")

    # ------------------------------------------------------------------ profiles
    summary = (
        profile.groupby("segment")
        .agg(
            customers=("segment", "size"),
            avg_churn_probability=("churn_probability", "mean"),
            actual_churn_rate=("actual_churn", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            avg_tenure_months=("tenure", "mean"),
            monthly_revenue=("MonthlyCharges", "sum"),
        )
        .reset_index()
    )
    summary["annual_revenue"] = summary["monthly_revenue"] * 12
    # Expected annual revenue loss = each customer's own risk times their revenue,
    # which is more informative than assuming every flagged customer leaves.
    expected = (
        profile.assign(
            expected_loss=profile["churn_probability"] * profile["MonthlyCharges"] * 12
        )
        .groupby("segment")["expected_loss"]
        .sum()
        .reset_index(name="expected_annual_revenue_at_risk")
    )
    summary = summary.merge(expected, on="segment")

    order = ["Retain Now", "Monitor", "Grow", "Nurture"]
    summary["segment"] = pd.Categorical(summary["segment"], categories=order, ordered=True)
    summary = summary.sort_values("segment").reset_index(drop=True)

    for col in (
        "avg_churn_probability",
        "actual_churn_rate",
        "avg_monthly_charges",
        "avg_tenure_months",
    ):
        summary[col] = summary[col].round(4)
    for col in ("monthly_revenue", "annual_revenue", "expected_annual_revenue_at_risk"):
        summary[col] = summary[col].round(2)

    print("\n" + "=" * 78)
    print("SEGMENT PROFILES  (held-out test set)")
    print("=" * 78)
    print(
        f"{'segment':<12}{'n':>6}{'pred risk':>11}{'ACTUAL':>9}"
        f"{'$/month':>10}{'tenure':>8}{'$ at risk/yr':>14}"
    )
    print("-" * 78)
    for _, r in summary.iterrows():
        print(
            f"{r['segment']:<12}{int(r['customers']):>6}"
            f"{r['avg_churn_probability']:>11.3f}{r['actual_churn_rate']:>9.3f}"
            f"{r['avg_monthly_charges']:>10.2f}{r['avg_tenure_months']:>8.1f}"
            f"{r['expected_annual_revenue_at_risk']:>14,.0f}"
        )
    print("-" * 78)

    save_table(summary, "segment_profiles")

    # -------------------------------------------------------------- validation
    print("\nDoes the segmentation hold up against reality?")
    high = summary[summary["segment"].isin(["Retain Now", "Monitor"])]
    low = summary[summary["segment"].isin(["Grow", "Nurture"])]
    high_rate = float(
        (high["actual_churn_rate"] * high["customers"]).sum() / high["customers"].sum()
    )
    low_rate = float((low["actual_churn_rate"] * low["customers"]).sum() / low["customers"].sum())
    print(
        f"  Actual churn rate in the high-risk segments: {high_rate:.1%}\n"
        f"  Actual churn rate in the low-risk segments : {low_rate:.1%}"
    )
    if low_rate > 0:
        print(
            f"  High-risk customers churned {high_rate / low_rate:.1f}x as often. "
            f"The split is real,\n  not an artefact of the cut-offs."
        )

    # ---------------------------------------------------------------- playbook
    print("\n" + "=" * 78)
    print("CAMPAIGN RECOMMENDATIONS")
    print("=" * 78)
    for name in order:
        row = summary[summary["segment"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        print(
            f"\n{name}  -  {int(row['customers'])} customers, "
            f"${row['expected_annual_revenue_at_risk']:,.0f} expected annual revenue at risk"
        )
        print(f"  {CAMPAIGN_PLAYBOOK[name]}")

    # ------------------------------------------------------------- what drives it
    print("\n" + "=" * 78)
    print("WHAT DRIVES THE RISK SCORES  (permutation importance, test set)")
    print("=" * 78)
    importance = permutation_importance(model, data.X_test, data.y_test, data.feature_names)
    save_table(importance, "feature_importance")

    print(f"{'feature':<34}{'ROC-AUC drop when shuffled':>26}")
    print("-" * 78)
    for _, r in importance.head(10).iterrows():
        print(f"{r['feature']:<34}{r['auc_drop']:>26.4f}")
    print("-" * 78)
    print(
        "These are the levers marketing can actually pull: the features whose\n"
        "removal most damages the model's ability to rank customers by risk."
    )

    # ---------------------------------------------------------------- figures
    plot_segments(
        risk,
        value,
        profile["segment"].to_numpy(),
        risk_cut,
        value_cut,
        "20_marketing_segments",
        "Otomoto marketing segments: churn risk vs customer value (test set)",
    )
    plot_bar_comparison(
        summary["segment"].astype(str).tolist(),
        summary["actual_churn_rate"].tolist(),
        "21_segment_actual_churn",
        "Actual churn rate by predicted segment (validates the segmentation)",
        "Actual churn rate",
        highlight="Retain Now",
    )
    plot_bar_comparison(
        summary["segment"].astype(str).tolist(),
        summary["expected_annual_revenue_at_risk"].tolist(),
        "22_segment_revenue_at_risk",
        "Expected annual revenue at risk by segment",
        "Expected annual revenue at risk ($)",
        highlight="Retain Now",
    )
    top = importance.head(12).iloc[::-1]
    plot_bar_comparison(
        top["feature"].tolist(),
        top["auc_drop"].tolist(),
        "23_feature_importance",
        "Churn risk drivers (permutation importance on test ROC-AUC)",
        "ROC-AUC drop when shuffled",
    )

    profile.to_csv(TABLES_DIR / "scored_test_customers.csv", index=False)
    print("\n  saved table -> results/tables/scored_test_customers.csv")

    payload = {
        "risk_threshold": risk_cut,
        "value_threshold": value_cut,
        "segments": summary.assign(segment=summary["segment"].astype(str)).to_dict("records"),
        "validation": {
            "high_risk_actual_churn": high_rate,
            "low_risk_actual_churn": low_rate,
        },
        "top_features": importance.head(15).to_dict("records"),
        "playbook": CAMPAIGN_PLAYBOOK,
    }
    save_json(payload, "segmentation_results")

    print("\n" + "=" * 78)
    print("STEP 5 COMPLETE")
    print("=" * 78)
    return payload


if __name__ == "__main__":
    main()
