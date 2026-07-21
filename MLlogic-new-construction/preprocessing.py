"""Feature engineering for the secondary-sales price model.

Reproduces the pipeline developed in notebooks/notebook_fabian_refined.ipynb:
  1. build_raw_features    -- log target, floor/position features, drop unused columns
  2. encode_ordinal_and_onehot -- ordinal-encode energy_class/condition, one-hot bezirk/transit_line
  3. target-encode ortsteil   -- leakage-safe: out-of-fold for training, a fitted
     lookup table for anything encoded afterwards (evaluation or live inference)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config import CONDITION_MAP, ENERGY_CLASS_MAP, KFOLD_SPLITS, RANDOM_STATE, TARGET_ENCODING_SMOOTHING


def load_raw_data(path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_raw_features(df_sales: pd.DataFrame) -> pd.DataFrame:
    """Log-transforms the target and derives floor/position features, then drops
    columns that are never used as model inputs (identifiers, raw text superseded
    by derived columns, USD price columns, and price_per_m2_eur -- collinear with
    the target since price_per_m2_eur = price_eur / area_m2).
    """
    df = df_sales.copy()

    df["price_eur_log"] = np.log1p(df["price_eur"])

    df["is_top_floor"] = (df["floor"] == df["total_floors"]).astype(int)
    df["is_ground_floor"] = (df["floor"] == 0).astype(int)

    drop_cols = [
    "id",
    "date_listed",
    "project_id",
    "project_name",
    "developer",
    "property_type",
    "transit_station",
    "total_project_units",
    "possession_status",
    "payment_plan",
    "price_per_m2_eur",
    "price_usd",
    "price_per_m2_usd",
]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df



def encode_ordinal_and_onehot(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["energy_class_ordinal"] = df["energy_class"].map(ENERGY_CLASS_MAP)

    bezirk_dummies = pd.get_dummies(
        df["bezirk"],
        prefix="bezirk",
        drop_first=True
    )

    transit_dummies = pd.get_dummies(
        df["transit_line"],
        prefix="transit_line",
        drop_first=True
    )

    df = pd.concat(
        [df, bezirk_dummies, transit_dummies],
        axis=1
    )

    df = df.drop(
        columns=[
            "lat",
            "lon",
            "energy_class",
            "bezirk",
            "transit_line",
            "price_eur",
        ],
        errors="ignore"
    )

    return df


def target_encode_out_of_fold(
    ortsteil: pd.Series,
    target: pd.Series,
    n_splits: int = KFOLD_SPLITS,
    smoothing: float = TARGET_ENCODING_SMOOTHING,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Leakage-safe out-of-fold target encoding for TRAINING rows: each row is
    encoded using only the mean/count computed from the *other* folds, never its
    own, so the model never trains on a row's own price baked into its own feature.
    """
    global_mean = target.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    encoded = pd.Series(index=ortsteil.index, dtype=float)
    for fit_idx, holdout_idx in kf.split(ortsteil):
        fold_stats = target.iloc[fit_idx].groupby(ortsteil.iloc[fit_idx]).agg(["mean", "count"])
        fold_smoothed = (fold_stats["mean"] * fold_stats["count"] + global_mean * smoothing) / (
            fold_stats["count"] + smoothing
        )
        encoded.iloc[holdout_idx] = ortsteil.iloc[holdout_idx].map(fold_smoothed).fillna(global_mean).values

    return encoded


def fit_ortsteil_lookup(
    ortsteil: pd.Series, target: pd.Series, smoothing: float = TARGET_ENCODING_SMOOTHING
) -> tuple[pd.Series, float]:
    """Fits the ortsteil -> smoothed mean `price_eur_log` lookup table used for
    encoding any data *after* training -- evaluation sets or live API requests.
    Returns (lookup, global_mean); global_mean is the fallback for an ortsteil
    never seen during training.
    """
    global_mean = target.mean()
    stats = target.groupby(ortsteil).agg(["mean", "count"])
    smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return smoothed, global_mean


def apply_ortsteil_lookup(ortsteil: pd.Series, lookup: pd.Series, global_mean: float) -> pd.Series:
    """Applies an already-fitted ortsteil lookup table to new data."""
    return ortsteil.map(lookup).fillna(global_mean)
