"""
Data loading and preprocessing for the Otomoto customer dataset.

Design note
-----------
Preprocessing is deliberately identical for the baseline and every optimised
model. The assignment measures the effect of optimisation algorithms, so the
input representation is held fixed; otherwise an apparent "optimiser win" could
really be a preprocessing win.

The scaler is fitted on the training split only and then applied to validation
and test. Fitting on the full dataset would leak test-set distribution
information into training and inflate the reported scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    ID_COLUMN,
    MULTICLASS_FEATURES,
    RANDOM_SEED,
    RAW_DATA,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)


@dataclass
class Dataset:
    """Container for the fully prepared, split and scaled data."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    raw_test: pd.DataFrame  # untransformed test rows, for segment interpretation

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def summary(self) -> str:
        def dist(y: np.ndarray) -> str:
            return f"{y.mean():.2%} churn"

        return (
            f"features={self.n_features}\n"
            f"  train: {self.X_train.shape[0]:>5} rows ({dist(self.y_train)})\n"
            f"  val:   {self.X_val.shape[0]:>5} rows ({dist(self.y_val)})\n"
            f"  test:  {self.X_test.shape[0]:>5} rows ({dist(self.y_test)})"
        )


def load_raw() -> pd.DataFrame:
    """Read the CSV exactly as supplied, with no transformation."""
    return pd.read_csv(RAW_DATA)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair the two known defects in this dataset.

    1. `TotalCharges` arrives as text and contains 11 blank strings. Every one of
       those rows has `tenure == 0`, i.e. a customer who joined but has not yet
       been billed. Zero is therefore the factually correct value; a mean or
       median fill would invent thousands of dollars of history for brand-new
       customers and distort the tenure/charges relationship the model relies on.
    2. `customerID` is a unique identifier with no predictive content. Left in,
       it would add 7043 meaningless dimensions after encoding.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    never_billed = df["TotalCharges"].isna()
    assert (df.loc[never_billed, "tenure"] == 0).all(), (
        "Unexpected missing TotalCharges for a customer with tenure > 0 - "
        "the zero-fill justification no longer holds, investigate before proceeding."
    )
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = df.drop(columns=[ID_COLUMN])

    return df


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map the target and binary columns to 0/1, one-hot the rest."""
    df = df.copy()

    y = df[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    df = df.drop(columns=[TARGET])

    for column, mapping in BINARY_FEATURES.items():
        df[column] = df[column].map(mapping).astype("int8")

    df = pd.get_dummies(df, columns=MULTICLASS_FEATURES, drop_first=True, dtype="int8")

    return df, y


def prepare(verbose: bool = True) -> Dataset:
    """Run the full pipeline and return train/validation/test arrays."""
    raw = load_raw()
    cleaned = clean(raw)
    X_df, y = encode(cleaned)

    # Stratify both splits so the ~26.5% churn rate is preserved everywhere.
    # Without stratification the minority class can drift between splits and the
    # validation signal becomes unreliable.
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X_df,
        y,
        np.arange(len(X_df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    # Fit the scaler on training data only, then reuse it. Neural networks are
    # sensitive to input scale: TotalCharges spans 0-8685 while the dummy columns
    # are 0/1, and that disparity alone can stall plain gradient descent.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[CONTINUOUS_FEATURES] = scaler.fit_transform(X_train[CONTINUOUS_FEATURES])
    X_val[CONTINUOUS_FEATURES] = scaler.transform(X_val[CONTINUOUS_FEATURES])
    X_test[CONTINUOUS_FEATURES] = scaler.transform(X_test[CONTINUOUS_FEATURES])

    dataset = Dataset(
        X_train=X_train.to_numpy(dtype="float32"),
        X_val=X_val.to_numpy(dtype="float32"),
        X_test=X_test.to_numpy(dtype="float32"),
        y_train=y_train.to_numpy(dtype="float32"),
        y_val=y_val.to_numpy(dtype="float32"),
        y_test=y_test.to_numpy(dtype="float32"),
        feature_names=list(X_df.columns),
        scaler=scaler,
        raw_test=cleaned.iloc[idx_test].reset_index(drop=True),
    )

    if verbose:
        print("Prepared dataset:")
        print(dataset.summary())

    return dataset


if __name__ == "__main__":
    prepare()
