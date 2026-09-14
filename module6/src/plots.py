"""
Figure generation. All plots are written to results/figures as PNG.

Matplotlib runs on the non-interactive Agg backend so the scripts work headless
and never block waiting for a window to close.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from config import FIGURES_DIR

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved figure-> results/figures/{name}.png")


def plot_learning_curves(
    histories: Mapping[str, Mapping[str, Sequence[float]]],
    name: str,
    title: str,
    metric: str = "loss",
    val_metric: str | None = None,
    ylabel: str | None = None,
) -> None:
    """
    Overlay the training curves of several runs.

    This is the core evidence for an optimiser comparison: the final score says
    which model ended up better, the curve says how fast it got there and whether
    it was still improving or already diverging.
    """
    val_metric = val_metric or f"val_{metric}"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for label, hist in histories.items():
        if metric in hist:
            axes[0].plot(hist[metric], label=label, linewidth=1.4)
        if val_metric in hist:
            axes[1].plot(hist[val_metric], label=label, linewidth=1.4)

    axes[0].set_title("Training")
    axes[1].set_title("Validation")
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel(ylabel or metric.replace("_", " ").title())
    axes[1].legend(fontsize=7, loc="best")

    fig.suptitle(title, fontsize=11)
    _save(fig, name)


def plot_overfitting_gap(
    history: Mapping[str, Sequence[float]],
    name: str,
    title: str,
) -> None:
    """Train vs validation loss for a single run, to make any divergence obvious."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["loss"], label="Training loss", linewidth=1.6)
    ax.plot(history["val_loss"], label="Validation loss", linewidth=1.6)

    val = np.asarray(history["val_loss"])
    best = int(np.argmin(val))
    ax.axvline(best, color="grey", linestyle="--", linewidth=1)
    ax.annotate(
        f"best val loss\nepoch {best}",
        xy=(best, val[best]),
        xytext=(8, 14),
        textcoords="offset points",
        fontsize=7,
        color="dimgrey",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    _save(fig, name)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Retained", "Churned"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    _save(fig, name)


def plot_roc(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]],
    aucs: Mapping[str, float],
    name: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for label, (y_true, proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{label} (AUC={aucs[label]:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, name)


def plot_bar_comparison(
    labels: Sequence[str],
    values: Sequence[float],
    name: str,
    title: str,
    ylabel: str,
    highlight: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6.0, 0.85 * len(labels) + 2), 4))
    colors = [
        "#c44e52" if highlight and lab == highlight else "#4c72b0" for lab in labels
    ]
    bars = ax.bar(labels, values, color=colors)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.25 or 0.01
    ax.set_ylim(max(0.0, lo - pad), hi + pad)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save(fig, name)


def plot_threshold_sweep(
    thresholds: np.ndarray,
    curves: Mapping[str, np.ndarray],
    chosen: float,
    name: str,
    title: str,
) -> None:
    """Show how precision, recall and F1 trade off as the cut-off moves."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in curves.items():
        ax.plot(thresholds, values, linewidth=1.6, label=label)

    ax.axvline(chosen, color="grey", linestyle="--", linewidth=1)
    ax.annotate(
        f"chosen = {chosen:.2f}",
        xy=(chosen, 0.05),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8,
        color="dimgrey",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    _save(fig, name)


def plot_segments(
    risk: np.ndarray,
    value: np.ndarray,
    segment: np.ndarray,
    risk_cut: float,
    value_cut: float,
    name: str,
    title: str,
) -> None:
    """Risk vs value scatter, coloured by assigned marketing segment."""
    fig, ax = plt.subplots(figsize=(6.4, 5))

    palette = {
        "Retain Now": "#c44e52",
        "Monitor": "#dd8452",
        "Grow": "#4c72b0",
        "Nurture": "#55a868",
    }
    for label, color in palette.items():
        mask = segment == label
        if mask.any():
            ax.scatter(
                risk[mask],
                value[mask],
                s=11,
                alpha=0.55,
                color=color,
                label=f"{label} (n={int(mask.sum())})",
                edgecolors="none",
            )

    ax.axvline(risk_cut, color="grey", linestyle="--", linewidth=1)
    ax.axhline(value_cut, color="grey", linestyle="--", linewidth=1)

    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Monthly charges (value proxy)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    _save(fig, name)
