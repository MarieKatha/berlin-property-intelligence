"""Prediction helper for the model bundle produced by train.py.

This is the piece an API imports: given the raw fields of a single listing
(the same columns as raw_data/secondary_sales.csv, minus price_eur -- that's
what we're predicting), it builds the exact feature vector the model expects
and returns a predicted price in EUR.

By default every field in REQUIRED_FIELDS must be provided. Pass
fill_missing=True to allow partial listings -- any feature that can't be
derived from what was provided is filled with its training-set average
instead (see the accuracy caveat on predict_price_eur below).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config import CONDITION_MAP, ENERGY_CLASS_MAP, MODEL_OUTPUT_PATH
from preprocessing import apply_ortsteil_lookup

REQUIRED_FIELDS = [
    "ortsteil", "rooms", "area_m2", "floor", "total_floors", "year_built",
    "energy_class", "condition", "has_lift", "has_balcony", "has_cellar",
    "has_parking", "transit_distance_min",
    "mortgage_rate_at_listing", "position",
]

# raw listing field -> feature column name, for the ones that pass straight through
_DIRECT_FIELDS = {
    "rooms": "rooms",
    "area_m2": "area_m2",
    "floor": "floor",
    "year_built": "year_built",
    "has_lift": "has_lift",
    "has_balcony": "has_balcony",
    "has_cellar": "has_cellar",
    "has_parking": "has_parking",
    "transit_distance_min": "transit_distance_min",
    "mortgage_rate_at_listing": "mortgage_rate_at_listing",
}

# raw listing field -> its one-hot feature-column prefix
_ONE_HOT_FIELDS = ("position",)


def load_model_bundle(path=MODEL_OUTPUT_PATH) -> dict:
    """Loads the dict saved by train.py: {model, feature_columns,
    feature_fill_values, ortsteil_lookup, ortsteil_global_mean}."""
    return joblib.load(path)


def _build_feature_row(listing: dict, bundle: dict, fill_missing: bool) -> pd.DataFrame:
    missing = [f for f in REQUIRED_FIELDS if f not in listing]
    if missing and not fill_missing:
        raise ValueError(
            f"listing is missing required fields: {missing}. "
            "Pass fill_missing=True to predict_price_eur() to fill them with "
            "training-set averages instead (see its docstring for the accuracy caveat)."
        )

    feature_columns = bundle["feature_columns"]
    row = pd.Series(index=feature_columns, dtype=float)

    for raw_field, col in _DIRECT_FIELDS.items():
        if raw_field in listing:
            row[col] = float(listing[raw_field])

    if "floor" in listing and "total_floors" in listing:
        row["is_top_floor"] = int(listing["floor"] == listing["total_floors"])
    if "floor" in listing:
        row["is_ground_floor"] = int(listing["floor"] == 0)

    if "energy_class" in listing:
        if listing["energy_class"] not in ENERGY_CLASS_MAP:
            raise ValueError(f"unknown energy_class: {listing['energy_class']!r}")
        row["energy_class_ordinal"] = ENERGY_CLASS_MAP[listing["energy_class"]]

    if "condition" in listing:
        if listing["condition"] not in CONDITION_MAP:
            raise ValueError(f"unknown condition: {listing['condition']!r}")
        row["condition_ordinal"] = CONDITION_MAP[listing["condition"]]

    if "ortsteil" in listing:
        lookup = apply_ortsteil_lookup(
            pd.Series([listing["ortsteil"]]), bundle["ortsteil_lookup"], bundle["ortsteil_global_mean"]
        )
        row["ortsteil_target_enc"] = lookup.iloc[0]

    for raw_field in _ONE_HOT_FIELDS:
        group_cols = [c for c in feature_columns if c.startswith(f"{raw_field}_")]
        if raw_field in listing:
            # Category known -> every dummy in this group is now determined: 1 for
            # a match, 0 otherwise. 0 also correctly represents an unseen category
            # or the drop_first=True reference category (standard one-hot fallback).
            dummy_col = f"{raw_field}_{listing[raw_field]}"
            for c in group_cols:
                row[c] = 1.0 if c == dummy_col else 0.0
        # else: raw_field entirely absent from the listing -> leave this group's
        # dummies as NaN, to be mean-filled below (fill_missing=True only).

    if fill_missing:
        row = row.fillna(bundle["feature_fill_values"])
    elif row.isna().any():
        # Shouldn't happen given the REQUIRED_FIELDS check above -- this is a
        # defensive guard in case REQUIRED_FIELDS and feature_columns ever drift
        # out of sync with each other.
        unresolved = row[row.isna()].index.tolist()
        raise ValueError(f"could not derive feature(s) {unresolved} from the listing")

    return row.to_frame().T[feature_columns]


def predict_price_eur(listing: dict, bundle: dict | None = None, fill_missing: bool = False) -> float:
    """Predicts the sale price in EUR for a single listing (a dict of raw fields,
    see REQUIRED_FIELDS).

    Pass an already-loaded `bundle` (from load_model_bundle) to avoid re-reading
    the .pkl on every call -- e.g. in an API, load it once at startup and reuse
    it per request.

    `fill_missing`: if False (default), every field in REQUIRED_FIELDS must be
    present or a ValueError is raised. If True, any missing field's feature(s)
    are filled with their training-set average instead of raising.

    Accuracy caveat: the notebook's "Robustness check" section (see
    notebooks/notebook_fabian_refined.ipynb) tested exactly this scenario for
    this model -- reusing the fully-trained XGBoost model with most features
    mean-filled degrades it substantially (test MAE went from ~€28,773 with all
    35 real features to ~€69,731 with only ortsteil/area_m2/condition known).
    fill_missing=True is a
    convenience for genuinely partial requests, not a substitute for providing
    real data -- the more fields you fill in, the more the prediction should be
    trusted.
    """
    if bundle is None:
        bundle = load_model_bundle()

    X_row = _build_feature_row(listing, bundle, fill_missing)
    price_eur_log_pred = bundle["model"].predict(X_row)[0]
    return float(np.expm1(price_eur_log_pred))


if __name__ == "__main__":
    example_listing = {
        "ortsteil": "Kreuzberg",
        "rooms": 2,
        "area_m2": 69.0,
        "floor": 1,
        "total_floors": 6,
        "year_built": 2011,
        "energy_class": "B",
        "condition": "saniert",
        "has_lift": True,
        "has_balcony": True,
        "has_cellar": False,
        "has_parking": False,
        "transit_distance_min": 10,
        "mortgage_rate_at_listing": 3.5,
        "position": "hinterhaus",
    }
    predicted_price = predict_price_eur(example_listing)
    print(f"Full listing:    €{predicted_price:,.0f}")

    partial_listing = {
        "ortsteil": "Kreuzberg",
        "area_m2": 69.0,
        "condition": "saniert",
    }
    predicted_price_partial = predict_price_eur(partial_listing, fill_missing=True)
    print(f"Partial listing: €{predicted_price_partial:,.0f}  (fill_missing=True)")
