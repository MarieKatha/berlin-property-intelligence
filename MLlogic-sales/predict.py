"""Prediction helper for the model bundle produced by train.py.

This is the piece an API imports: given the raw fields of a single listing
(the same columns as raw_data/secondary_sales.csv, minus price_eur -- that's
what we're predicting), it builds the exact feature vector the model expects
and returns a predicted price in EUR.

By default every field in REQUIRED_FIELDS must be provided. Pass
fill_missing=True to allow partial listings -- any feature that can't be
derived from what was provided is filled with its training-set average
instead (see the accuracy caveat on predict_price_eur below).

load_explainer/explain_prediction add SHAP-based explanations on top of a
prediction -- see explain_prediction's docstring.
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


def _fix_xgboost_base_score_for_shap(booster) -> None:
    """Patches `booster.save_raw` so shap.TreeExplainer can read this model.

    xgboost>=3.0 serializes a single-target regressor's base_score as a
    bracketed string, e.g. "[12.372406]", in its UBJSON dump. shap<0.50
    expects a plain numeric string there and raises `ValueError: could not
    convert string to float` on TreeExplainer construction otherwise --
    shap>=0.50 (needs Python>=3.11, see requirements.txt) parses the
    bracketed form fine on its own. This patch strips the brackets from that
    one field, byte-for-byte, in the UBJSON buffer shap reads the model
    through, leaving every other byte untouched -- it's a no-op (returns the
    buffer unchanged) whenever the format doesn't match what it's looking
    for, so it's safe to apply unconditionally regardless of shap version.
    Only needed for shap<0.50; kept as a defensive fallback since this
    project runs across multiple Python versions (3.10-3.12) with different
    shap versions actually installable.
    """
    original_save_raw = booster.save_raw

    def patched_save_raw(raw_format="deprecated"):
        raw = original_save_raw(raw_format=raw_format)
        if raw_format != "ubj":
            return raw
        raw = bytes(raw)
        key = b"base_score"
        idx = raw.find(key)
        if idx == -1:
            return raw
        pos = idx + len(key)
        if raw[pos:pos + 1] != b"S":  # not a string-encoded base_score -- nothing to fix
            return raw
        pos += 1
        length_type = raw[pos:pos + 1]
        length_sizes = {b"i": 1, b"U": 1, b"I": 2, b"l": 4, b"L": 8}
        if length_type not in length_sizes:
            return raw
        length_size = length_sizes[length_type]
        signed = length_type in (b"i", b"I", b"l", b"L")
        old_length = int.from_bytes(raw[pos + 1:pos + 1 + length_size], "big", signed=signed)
        content_start = pos + 1 + length_size
        old_content = raw[content_start:content_start + old_length]
        if not (old_content.startswith(b"[") and old_content.endswith(b"]")):
            return raw  # already plain, e.g. already patched, or a shap version that stored it differently
        new_content = old_content[1:-1]
        new_length_bytes = len(new_content).to_bytes(length_size, "big", signed=signed)
        return raw[:pos + 1] + new_length_bytes + new_content + raw[content_start + old_length:]

    booster.save_raw = patched_save_raw


def load_explainer(bundle: dict):
    """Builds a SHAP TreeExplainer for the model in `bundle`, for use with
    explain_prediction below. Construction takes a moment (it parses every
    tree) -- build once, e.g. at API startup, and reuse it across requests
    rather than rebuilding per call.

    Requires the optional `shap` dependency -- imported lazily here so the
    rest of this module (predict_price_eur etc.) works fine without it.
    """
    import shap

    _fix_xgboost_base_score_for_shap(bundle["model"].get_booster())
    return shap.TreeExplainer(bundle["model"])


# Maps each raw listing field to the model feature column(s) it controls, for
# explain_prediction below. is_top_floor/is_ground_floor are derived jointly
# from floor+total_floors (see _build_feature_row): is_ground_floor only
# needs floor, so it's grouped under "floor"; is_top_floor needs both, so its
# contribution is attributed to "total_floors" specifically (floor's own
# direct effect is already covered by the "floor" and "is_ground_floor"
# columns). "position" isn't listed here -- it's a one-hot group whose column
# names depend on the chosen category, so it's resolved dynamically instead.
_FIELD_TO_FEATURE_COLUMNS = {
    "ortsteil": ["ortsteil_target_enc"],
    "area_m2": ["area_m2"],
    "condition": ["condition_ordinal"],
    "rooms": ["rooms"],
    "floor": ["floor", "is_ground_floor"],
    "total_floors": ["is_top_floor"],
    "year_built": ["year_built"],
    "energy_class": ["energy_class_ordinal"],
    "has_lift": ["has_lift"],
    "has_balcony": ["has_balcony"],
    "has_cellar": ["has_cellar"],
    "has_parking": ["has_parking"],
    "transit_distance_min": ["transit_distance_min"],
    "mortgage_rate_at_listing": ["mortgage_rate_at_listing"],
}


def explain_prediction(listing: dict, bundle: dict, explainer) -> tuple[float, list[dict]]:
    """Predicts the price like predict_price_eur (always with
    fill_missing=True), and additionally returns how much each EXPLICITLY
    PROVIDED field in `listing` moved that prediction, in EUR, sorted by
    |impact| descending.

    A field not in `listing` -- filled with its training-set average -- is
    left out of the breakdown entirely: its SHAP contribution would describe
    a "typical" listing rather than anything the caller actually told us, so
    surfacing it would be misleading.

    Uses SHAP TreeExplainer, which computes exact Shapley values for tree
    models directly from the trained model, no retraining needed. Because
    this model predicts log1p(price), not price itself, SHAP values are
    additive in log-space, not EUR. Converting a feature's contribution to a
    EUR figure is therefore an approximation: for each field, it's the
    difference between the actual prediction and the counterfactual
    prediction with just that field's SHAP contribution removed. These
    per-field EUR figures won't sum exactly to (predicted price - "nothing
    known" baseline) because of that log/EUR nonlinearity, but each one is
    individually a faithful answer to "how much is this field worth here".
    """
    X_row = _build_feature_row(listing, bundle, fill_missing=True)
    shap_row = np.asarray(explainer.shap_values(X_row)).reshape(-1)
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    log_pred = base_value + shap_row.sum()
    predicted_price_eur = float(np.expm1(log_pred))

    columns = list(X_row.columns)
    col_index = {c: i for i, c in enumerate(columns)}

    explanations = []
    for field, value in listing.items():
        if field == "position":
            feature_cols = [c for c in columns if c.startswith("position_")]
        else:
            feature_cols = _FIELD_TO_FEATURE_COLUMNS.get(field)
        if not feature_cols:
            continue  # not a model input this function knows how to attribute
        field_shap = sum(shap_row[col_index[c]] for c in feature_cols if c in col_index)
        counterfactual_price = float(np.expm1(log_pred - field_shap))
        explanations.append({
            "field": field,
            "value": value,
            "impact_eur": round(predicted_price_eur - counterfactual_price, 2),
        })

    explanations.sort(key=lambda e: abs(e["impact_eur"]), reverse=True)
    return predicted_price_eur, explanations


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

    bundle = load_model_bundle()
    explainer = load_explainer(bundle)
    predicted_price, explanation = explain_prediction(partial_listing, bundle, explainer)
    print(f"\nExplanation for partial listing (€{predicted_price:,.0f}):")
    for item in explanation:
        print(f"  {item['field']:<12} = {item['value']!r:<15} impact: {item['impact_eur']:+,.2f} EUR")
