"""Prediction helper for the model bundle produced by train.py.

This is the piece an API imports: given the raw fields of a single listing
(the same columns as raw_data/new_construction.csv, minus price_eur -- that's
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

from config import ENERGY_CLASS_MAP, MODEL_OUTPUT_PATH
from preprocessing import apply_ortsteil_lookup

REQUIRED_FIELDS = [
    "ortsteil",
    "bezirk",
    "rooms",
    "area_m2",
    "floor",
    "total_floors",
    "completion_year",
    "energy_class",
    "has_lift",
    "has_balcony",
    "has_parking",
    "transit_line",
    "transit_distance_min",
    "mortgage_rate_at_listing",
]

# raw listing field -> feature column name, for the ones that pass straight through
_DIRECT_FIELDS = {
    "rooms": "rooms",
    "area_m2": "area_m2",
    "floor": "floor",
    "total_floors": "total_floors",
    "completion_year": "completion_year",
    "has_lift": "has_lift",
    "has_balcony": "has_balcony",
    "has_parking": "has_parking",
    "transit_distance_min": "transit_distance_min",
    "mortgage_rate_at_listing": "mortgage_rate_at_listing",
}

# raw listing field -> its one-hot feature-column prefix
_ONE_HOT_FIELDS = ("bezirk", "transit_line")

# Model features that were trained on but can't be derived from any field a
# caller could plausibly provide (to_brandenburg_gate_km would need a real
# geocoder). These always fall back to 0, in strict mode too -- matches this
# model's pre-fill_missing behaviour, which silently zero-filled the same gap
# via reindex(..., fill_value=0).
_UNDERIVABLE_FEATURES = ("to_brandenburg_gate_km",)


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

    for col in _UNDERIVABLE_FEATURES:
        row[col] = 0.0

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
    shap versions actually installable. (Same fix as MLlogic-sales/predict.py
    -- duplicated rather than shared, matching this project's pattern of
    keeping each MLlogic-* folder self-contained.)
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
            return raw
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
# explain_prediction below. is_ground_floor is derived from floor alone;
# is_top_floor needs both floor and total_floors, and total_floors is ALSO a
# feature in its own right here (unlike the sales model) -- so it's grouped
# under "total_floors" along with is_top_floor, while "floor" covers its own
# direct column plus is_ground_floor. "ortsteil" is intentionally not paired
# with its one-hot dummies here -- "bezirk" isn't a request parameter at all
# (api/fast.py derives it automatically from ortsteil), so its contribution
# is merged into "ortsteil"'s below rather than reported separately or
# silently dropped. "transit_line" is a one-hot group resolved dynamically.
_FIELD_TO_FEATURE_COLUMNS = {
    "ortsteil": ["ortsteil_target_enc"],
    "area_m2": ["area_m2"],
    "rooms": ["rooms"],
    "floor": ["floor", "is_ground_floor"],
    "total_floors": ["total_floors", "is_top_floor"],
    "completion_year": ["completion_year"],
    "energy_class": ["energy_class_ordinal"],
    "has_lift": ["has_lift"],
    "has_balcony": ["has_balcony"],
    "has_parking": ["has_parking"],
    "transit_distance_min": ["transit_distance_min"],
    "mortgage_rate_at_listing": ["mortgage_rate_at_listing"],
}


def explain_prediction(listing: dict, bundle: dict, explainer) -> tuple[float, list[dict]]:
    """Predicts the price like predict_price_eur (always with
    fill_missing=True), and additionally returns how much each EXPLICITLY
    PROVIDED field in `listing` moved that prediction, in EUR, sorted by
    |impact| descending.

    Same idea as MLlogic-sales/predict.py's explain_prediction -- see its
    docstring for the full explanation of the SHAP TreeExplainer approach
    and the log-price/EUR counterfactual approximation this also uses (this
    model predicts price_eur_log too, not price directly). One difference:
    `bezirk` is never a request field here (it's auto-derived from
    `ortsteil`, see api/fast.py), so its SHAP contribution is folded into
    `ortsteil`'s reported impact rather than dropped or shown separately.
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
        if field == "bezirk":
            continue  # folded into "ortsteil" below, not reported on its own
        if field == "ortsteil":
            feature_cols = _FIELD_TO_FEATURE_COLUMNS["ortsteil"] + [c for c in columns if c.startswith("bezirk_")]
        elif field == "transit_line":
            # A recognized model input -- included even if this resolves to
            # zero columns, so it'd be an explicit 0.0 impact rather than
            # silently missing from the response.
            feature_cols = [c for c in columns if c.startswith("transit_line_")]
        elif field in _FIELD_TO_FEATURE_COLUMNS:
            feature_cols = _FIELD_TO_FEATURE_COLUMNS[field]
        else:
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

    Accuracy caveat: filling missing features with their training-set average
    degrades the model -- the fewer fields you actually provide, the less the
    prediction should be trusted (see the equivalent caveat in
    MLlogic-sales/predict.py, which this mirrors).
    """
    if bundle is None:
        bundle = load_model_bundle()

    X_row = _build_feature_row(listing, bundle, fill_missing)
    price_eur_log_pred = bundle["model"].predict(X_row)[0]
    return float(np.expm1(price_eur_log_pred))


if __name__ == "__main__":
    example_listing = {
        "ortsteil": "Mitte (Ort)",
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
    print(f"Full listing:    €{predicted_price:,.0f}")

    partial_listing = {
        "ortsteil": "Mitte (Ort)",
        "bezirk": "Mitte",
        "area_m2": 65,
    }
    predicted_price_partial = predict_price_eur(partial_listing, fill_missing=True)
    print(f"Partial listing: €{predicted_price_partial:,.0f}  (fill_missing=True)")

    bundle = load_model_bundle()
    explainer = load_explainer(bundle)
    predicted_price, explanation = explain_prediction(partial_listing, bundle, explainer)
    print(f"\nExplanation for partial listing (€{predicted_price:,.0f}):")
    for item in explanation:
        print(f"  {item['field']:<12} = {item['value']!r:<15} impact: {item['impact_eur']:+,.2f} EUR")
