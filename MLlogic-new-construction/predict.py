"""Prediction helper for the model bundle produced by train.py.

This is the piece an API imports: given the raw fields of a single listing
(the same columns as raw_data/secondary_sales.csv, minus price_eur -- that's
what we're predicting), it builds the exact feature vector the model expects
and returns a predicted price in EUR.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import CONDITION_MAP, ENERGY_CLASS_MAP, MODEL_OUTPUT_PATH
from preprocessing import apply_ortsteil_lookup

REQUIRED_FIELDS = [
    "ortsteil",
    "bezirk",
    "rooms",
    "area_m2",
    "floor",
    "total_floors",
    "completion_year",
    "has_lift",
    "has_balcony",
    "has_parking",
    "transit_line",
    "transit_distance_min",
    "mortgage_rate_at_listing",
]


def load_model_bundle(path=MODEL_OUTPUT_PATH) -> dict:
    """Loads the dict saved by train.py: {model, feature_columns, ortsteil_lookup,
    ortsteil_global_mean}."""
    return joblib.load(path)


def _build_feature_row(listing: dict, bundle: dict) -> pd.DataFrame:
    missing = [f for f in REQUIRED_FIELDS if f not in listing]
    if missing:
        raise ValueError(f"listing is missing required fields: {missing}")

    row = pd.DataFrame([listing])

    row["is_top_floor"] = int(row.loc[0, "floor"] == row.loc[0, "total_floors"])
    row["is_ground_floor"] = int(row.loc[0, "floor"] == 0)

    row["energy_class_ordinal"] = row["energy_class"].map(ENERGY_CLASS_MAP)

    if row["energy_class_ordinal"].isna().any():
        raise ValueError(f"unknown energy_class: {listing['energy_class']!r}")


    row["ortsteil_target_enc"] = apply_ortsteil_lookup(
        row["ortsteil"], bundle["ortsteil_lookup"], bundle["ortsteil_global_mean"]
    )

    # One-hot columns (position_*, bezirk_*, transit_line_*): set the one matching
    # this listing's category to 1. Any category that isn't a training-time dummy
    # column (either unseen, or the drop_first=True reference category) simply
    # leaves every dummy at 0 via the reindex below -- correct one-hot semantics.
    for col in ("bezirk", "transit_line"):
        dummy_col = f"{col}_{listing[col]}"
        row[dummy_col] = 1

    X_row = row.reindex(columns=bundle["feature_columns"], fill_value=0)
    return X_row


def predict_price_eur(listing: dict, bundle: dict | None = None) -> float:
    """Predicts the sale price in EUR for a single listing (a dict of raw fields,
    see REQUIRED_FIELDS). Pass an already-loaded `bundle` (from load_model_bundle)
    to avoid re-reading the .pkl on every call, e.g. in an API load it once at
    startup and reuse it per request.
    """
    if bundle is None:
        bundle = load_model_bundle()

    X_row = _build_feature_row(listing, bundle)
    price_eur_log_pred = bundle["model"].predict(X_row)[0]
    return float(np.expm1(price_eur_log_pred))


if __name__ == "__main__":
    example_listing = {
    "ortsteil": "Mitte",
    "bezirk": "Mitte",
    "rooms": 2,
    "area_m2": 65,
    "floor": 3,
    "total_floors": 6,
    "completion_year": 2026,
    "energy_class": "A",
    "has_lift": True,
    "has_balcony": True,
    "has_parking": False,
    "transit_line": "U2",
    "transit_distance_min": 5,
    "mortgage_rate_at_listing": 3.5,
}

    predicted_price = predict_price_eur(example_listing)
    print(f"Predicted price: €{predicted_price:,.0f}")
