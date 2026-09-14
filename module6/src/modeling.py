"""
Shared model-building, training and evaluation utilities.

Every experiment in this project goes through these functions so that the only
thing varying between runs is the factor under test (optimiser, learning rate,
batch size, architecture, regularisation).
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence

# Silence TensorFlow's C++ info logs before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

import keras
from keras import layers, regularizers
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import MAX_EPOCHS, RANDOM_SEED, TABLES_DIR


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds(seed: int = RANDOM_SEED) -> None:
    """
    Seed every RNG that can influence training.

    Weight initialisation, shuffling and dropout masks are all stochastic. Without
    this, two runs of the "same" configuration differ, and a small optimiser gap
    becomes indistinguishable from noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """A complete, reproducible description of one model configuration."""

    name: str
    hidden_units: Sequence[int] = (16,)
    activation: str = "relu"
    dropout: float = 0.0
    l2: float = 0.0
    batch_norm: bool = False
    optimizer: str = "sgd"
    learning_rate: float = 0.01
    momentum: float = 0.0
    loss: str = "binary_crossentropy"
    batch_size: int = 32
    epochs: int = MAX_EPOCHS
    early_stopping: bool = False
    patience: int = 15
    class_weight: bool = False
    notes: str = ""

    def describe(self) -> dict[str, Any]:
        return asdict(self)


def make_optimizer(spec: ModelSpec) -> keras.optimizers.Optimizer:
    """
    Instantiate the optimiser named in the spec.

    These are the algorithms from the module notes. The distinction that matters:
    plain SGD applies one global learning rate to every parameter, Momentum adds a
    velocity term to damp oscillation, AdaGrad/RMSProp/Adam adapt the step size
    per parameter from the history of squared gradients.
    """
    key = spec.optimizer.lower()

    if key == "sgd":
        # momentum=0.0 makes this textbook mini-batch gradient descent.
        return keras.optimizers.SGD(learning_rate=spec.learning_rate, momentum=0.0)
    if key == "momentum":
        return keras.optimizers.SGD(
            learning_rate=spec.learning_rate, momentum=spec.momentum or 0.9
        )
    if key == "nesterov":
        return keras.optimizers.SGD(
            learning_rate=spec.learning_rate,
            momentum=spec.momentum or 0.9,
            nesterov=True,
        )
    if key == "adagrad":
        return keras.optimizers.Adagrad(learning_rate=spec.learning_rate)
    if key == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=spec.learning_rate)
    if key == "adam":
        return keras.optimizers.Adam(learning_rate=spec.learning_rate)
    if key == "nadam":
        return keras.optimizers.Nadam(learning_rate=spec.learning_rate)

    raise ValueError(f"Unknown optimizer: {spec.optimizer!r}")


def safe_layer_name(name: str) -> str:
    """
    Turn a human-readable spec name into a legal TensorFlow scope name.

    TF requires ^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$, so descriptive names like
    "Baseline (SGD, lr=0.01)" are rejected for the parentheses, comma and equals
    sign. The readable name is kept on the spec for reports; only the graph-level
    name is sanitised.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    if not cleaned or not re.match(r"^[A-Za-z0-9.]", cleaned):
        cleaned = f"model_{cleaned}" if cleaned else "model"
    return cleaned


def build_model(spec: ModelSpec, n_features: int) -> keras.Model:
    """
    Assemble a feed-forward binary classifier from the spec.

    The output layer is a single sigmoid unit, which pairs with binary
    cross-entropy to produce a calibrated churn probability rather than a bare
    class label. Marketing needs the probability: it is what allows ranking
    customers by risk and setting a campaign budget threshold.
    """
    set_seeds()

    kernel_reg = regularizers.l2(spec.l2) if spec.l2 > 0 else None

    model = keras.Sequential(name=safe_layer_name(spec.name))
    model.add(layers.Input(shape=(n_features,)))

    for units in spec.hidden_units:
        model.add(layers.Dense(units, activation=None, kernel_regularizer=kernel_reg))
        # Batch norm before the activation is the original formulation and keeps
        # the pre-activation distribution stable across layers.
        if spec.batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation(spec.activation))
        if spec.dropout > 0:
            model.add(layers.Dropout(spec.dropout))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=make_optimizer(spec),
        loss=spec.loss,
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    """Everything produced by one training run."""

    spec: ModelSpec
    model: keras.Model
    history: dict[str, list[float]]
    train_seconds: float
    epochs_run: int
    metrics: dict[str, float] = field(default_factory=dict)


def compute_class_weight(y: np.ndarray) -> dict[int, float]:
    """
    Inverse-frequency weights for the imbalanced target.

    With ~73% non-churners, an unweighted model can score 73% accuracy by
    predicting "no churn" for everyone - and be useless for a retention campaign,
    because it never flags anyone. Weighting raises the cost of missing a churner.
    """
    n = len(y)
    n_pos = float(y.sum())
    n_neg = n - n_pos
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}


def train_model(spec: ModelSpec, data, verbose: int = 0) -> RunResult:
    """Fit one configuration and return the model plus its training history."""
    model = build_model(spec, data.n_features)

    callbacks: list[keras.callbacks.Callback] = []
    if spec.early_stopping:
        # Monitor validation AUC rather than loss: AUC is threshold-free and
        # tracks ranking quality, which is what the segmentation depends on.
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=spec.patience,
                restore_best_weights=True,
                verbose=verbose,
            )
        )

    class_weight = compute_class_weight(data.y_train) if spec.class_weight else None

    start = time.perf_counter()
    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
    )
    elapsed = time.perf_counter() - start

    return RunResult(
        spec=spec,
        model=model,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        train_seconds=elapsed,
        epochs_run=len(history.history["loss"]),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def predict_proba(model: keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, verbose=0).ravel()


def evaluate(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Score a fitted model on one split.

    Accuracy alone is reported but never relied on: at a 26.5% positive rate the
    majority-class baseline already scores 73.5%. Recall (what share of true
    churners we catch), F1 and ROC-AUC carry the real signal, and log loss is the
    cross-entropy the optimiser was actually minimising.
    """
    proba = predict_proba(model, X)
    pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7))),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def best_threshold(model: keras.Model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Find the decision threshold that maximises F1 on a validation split.

    The default 0.5 cut-off assumes balanced classes and equal error costs.
    Neither holds here, so the threshold is treated as a tunable parameter -
    selected on validation data, never on test.
    """
    proba = predict_proba(model, X)
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in candidates]
    idx = int(np.argmax(scores))
    return float(candidates[idx]), float(scores[idx])


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"  saved table -> {path.relative_to(TABLES_DIR.parent.parent)}")


def save_json(payload: Any, name: str) -> None:
    path = TABLES_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  saved json  -> {path.relative_to(TABLES_DIR.parent.parent)}")


def results_frame(results: Iterable[RunResult], split_label: str = "validation") -> pd.DataFrame:
    """Flatten a set of runs into one comparison table."""
    rows = []
    for r in results:
        row = {"model": r.spec.name, "optimizer": r.spec.optimizer}
        row.update(r.metrics)
        row["epochs_run"] = r.epochs_run
        row["train_seconds"] = round(r.train_seconds, 2)
        row["split"] = split_label
        rows.append(row)
    return pd.DataFrame(rows)
