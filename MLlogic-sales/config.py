"""Shared configuration for the secondary-sales price model.

Encoding choices and hyperparameters mirror what was validated in
notebooks/notebook_fabian_refined.ipynb (target-encoded ortsteil, ordinal
energy_class/condition, tuned XGBoost). bezirk and transit_line are
deliberately excluded -- neither is used as a model input.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = REPO_ROOT / "raw_data" / "secondary_sales.csv"
MODEL_OUTPUT_PATH = Path(__file__).resolve().parent / "model.pkl"

TARGET = "price_eur_log"

# Ordinal encodings: natural best/worst ordering (see notebooks/notebook_fabian_refined.ipynb)
ENERGY_CLASS_MAP = {
    "A_plus": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
}  # 0 = most energy-efficient, 8 = least efficient
CONDITION_MAP = {
    "renovierungsbedürftig": 0, "renoviert": 1, "modernisiert": 2, "saniert": 3, "kernsaniert": 4,
}  # 0 = worst condition, 4 = best condition

# Best hyperparameters found via RandomizedSearchCV in notebook_fabian_refined.ipynb
# (test MAE ~€28,007, test R² ~0.974 on the held-out evaluation split)
#
# n_jobs=1 (not -1) is deliberate: XGBoost's multi-threaded histogram building
# sums floats in a different order depending on thread scheduling, so n_jobs=-1
# can produce a *slightly different* model on every training run even with a
# fixed random_state (observed ~1-2% prediction drift on the same input between
# runs). n_jobs=1 trades some training speed for byte-for-byte reproducibility,
# which matters more for a deployment artifact than for notebook experimentation.
XGB_PARAMS = {
    "n_estimators": 700,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 0.6,
    "min_child_weight": 5,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42,
    "n_jobs": 1,
}

TARGET_ENCODING_SMOOTHING = 10
KFOLD_SPLITS = 5
RANDOM_STATE = 42
