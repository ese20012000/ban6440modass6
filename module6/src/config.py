"""
Central configuration for the Otomoto marketing-segmentation ANN project.

Keeping paths, seeds and column groupings in one place means every experiment
script shares an identical data contract. That matters here: the assignment
compares optimisation algorithms, so preprocessing must be held constant or the
comparison is confounded.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA = DATA_DIR / "teleconnect.csv"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
RANDOM_SEED = 42

# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------
TARGET = "Churn"
ID_COLUMN = "customerID"

# Continuous features -> standardised. SeniorCitizen is already 0/1 so it is
# deliberately excluded from scaling.
CONTINUOUS_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Two-level categoricals -> simple 0/1 mapping (no need for a dummy column).
BINARY_FEATURES = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

# Three-plus-level categoricals -> one-hot encoded.
# "No internet service" / "No phone service" are kept as their own level because
# they are structurally different from a plain "No" (the customer cannot have
# the add-on at all, rather than declining it).
MULTICLASS_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

PASSTHROUGH_FEATURES = ["SeniorCitizen"]

# ----------------------------------------------------------------------------
# Split proportions (stratified on the target)
# ----------------------------------------------------------------------------
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # taken from the remaining 80% -> 64/16/20 overall

# ----------------------------------------------------------------------------
# Training defaults
# ----------------------------------------------------------------------------
MAX_EPOCHS = 150
DEFAULT_BATCH_SIZE = 32
