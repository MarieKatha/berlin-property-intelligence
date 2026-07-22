"""
Berlin Rental Price Prediction Model
Inference module for API / AI Agent
"""

import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


class RentalPricePredictor:

    def __init__(self):
        """
        Load trained model and preprocessing artifacts
        """

        # Directory where this file is located
        BASE_DIR = Path(__file__).resolve().parent

        # File locations
        model_path = BASE_DIR / "xgboost_rental_model.json"  # Changed to .json
        metadata_path = BASE_DIR / "metadata.json"

        target_encoder_path = BASE_DIR / "target_encoder.pkl"
        energy_encoder_path = BASE_DIR / "energy_encoder.pkl"
        condition_encoder_path = BASE_DIR / "condition_encoder.pkl"

        # Load model (JSON format)
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load encoders
        with open(target_encoder_path, "rb") as f:
            self.target_encoder = pickle.load(f)

        with open(energy_encoder_path, "rb") as f:
            self.energy_encoder = pickle.load(f)

        with open(condition_encoder_path, "rb") as f:
            self.condition_encoder = pickle.load(f)

        # Expected model feature order
        self.feature_columns = self.metadata["feature_columns"]


    # raw listing field -> its one-hot feature-column prefix
    _CATEGORICAL_FEATURES = ("bezirk", "transit_line", "position")

    def _build_feature_frame(self, input_data):
        """Builds the model-ready feature DataFrame -- same values as
        preprocess(), but keeping column names/order instead of collapsing to
        a plain numpy array, so explain() below can map SHAP values back to
        the raw fields that produced them. preprocess() is just this plus
        `.values`; kept as one shared implementation so the two never drift.
        """
        df = pd.DataFrame([input_data])

        # Target encoding
        df["ortsteil"] = self.target_encoder.transform(df[["ortsteil"]])

        # Ordinal encoding -- only if provided; a genuinely missing field is left
        # for the fill-in step below instead of KeyError-ing here.
        if "energy_class" in df.columns:
            # This encoder was fit on "A+", not the "A_plus" spelling used by the
            # raw data and every other endpoint in this project -- alias it so
            # callers can pass the same "A_plus" value everywhere.
            df["energy_class"] = df["energy_class"].replace({"A_plus": "A+"})
            df["energy_class"] = self.energy_encoder.transform(df[["energy_class"]])
        if "condition" in df.columns:
            df["condition"] = self.condition_encoder.transform(df[["condition"]])

        # One-hot encoding -- only for whichever of these were actually provided.
        # A category the caller omits entirely just leaves that group's dummies
        # unset, filled in as 0 below (same as an unseen/reference category).
        for col in self._CATEGORICAL_FEATURES:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)

        # Match training features. One-hot dummies default to 0 (not-this-category);
        # anything else missing (an ordinal/continuous feature the caller didn't
        # provide) is left as NaN, which this XGBoost model was trained to handle
        # natively (see metadata.json's "missing": NaN) -- more honest than
        # silently guessing 0 for a feature we simply don't know.
        dummy_prefixes = tuple(f"{col}_" for col in self._CATEGORICAL_FEATURES) + ("building_era_",)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0 if col.startswith(dummy_prefixes) else np.nan

        return df[self.feature_columns]

    def preprocess(self, input_data):
        """
        Convert raw property input into model-ready features

        Args:
            input_data (dict): Property characteristics

        Returns:
            numpy array: Features in correct model order
        """
        return self._build_feature_frame(input_data).values


    def predict(self, input_data):
        """
        Predict rental price

        Args:
            input_data (dict): Apartment characteristics

        Returns:
            dict: Prediction + model performance
        """

        X = self.preprocess(input_data)
        prediction = self.model.predict(X)[0]

        return {
        'predicted_rent_eur': float(prediction),
        'model_type': self.metadata["model"]["type"],
        'confidence': {
        'r2_score': self.metadata["performance"]["r2"],
        'mae': self.metadata["performance"]["mae"],
        'rmse': self.metadata["performance"]["rmse"]
    }
}

    def explain(self, input_data, explainer):
        """Predicts the rent like predict(), and additionally returns how
        much each EXPLICITLY PROVIDED field in `input_data` moved that
        prediction, in EUR, sorted by |impact| descending.

        Same idea as MLlogic-sales/predict.py's explain_prediction (SHAP
        TreeExplainer; fields not in `input_data` -- filled with a
        training-set-derived default -- are left out of the breakdown
        entirely, since their "contribution" would describe a typical
        listing, not this one). One difference: this model predicts
        warmmiete_eur_monthly directly, not a log-transformed target, so
        SHAP values ARE additive EUR contributions here -- no counterfactual
        expm1() approximation needed, unlike the sales/construction models.

        `bezirk` is never a request field (api/fast.py derives it
        automatically from `ortsteil`), so its SHAP contribution is folded
        into `ortsteil`'s reported impact rather than dropped or shown
        separately. `position` currently has zero effect on this model's
        predictions no matter what's passed -- see _build_feature_frame:
        its one-hot dummies are computed but aren't in self.feature_columns,
        so they never reach the model -- and will correctly show up with an
        impact of 0.0 if provided, not because of a bug in this function.
        """
        X = self._build_feature_frame(input_data)
        columns = list(X.columns)
        col_index = {c: i for i, c in enumerate(columns)}

        shap_row = np.asarray(explainer.shap_values(X.values)).reshape(-1)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        predicted_rent_eur = float(base_value + shap_row.sum())

        explanations = []
        for field, value in input_data.items():
            if field == "bezirk":
                continue  # folded into "ortsteil" below, not reported on its own
            if field == "ortsteil":
                feature_cols = ["ortsteil"] + [c for c in columns if c.startswith("bezirk_")]
            elif field in ("transit_line", "position"):
                # Recognized model inputs -- included even if this resolves to
                # zero columns (e.g. "position", see docstring), so it's an
                # explicit 0.0 impact rather than silently missing from the
                # response.
                feature_cols = [c for c in columns if c.startswith(f"{field}_")]
            elif field in _FIELD_TO_FEATURE_COLUMNS:
                feature_cols = _FIELD_TO_FEATURE_COLUMNS[field]
            else:
                continue  # not a model input this function knows how to attribute
            field_shap = sum(shap_row[col_index[c]] for c in feature_cols if c in col_index)
            explanations.append({
                "field": field,
                "value": value,
                "impact_eur": round(float(field_shap), 2),
            })

        explanations.sort(key=lambda e: abs(e["impact_eur"]), reverse=True)
        return predicted_rent_eur, explanations


# Maps each raw listing field to the model feature column(s) it controls, for
# RentalPricePredictor.explain() above. "ortsteil"/"transit_line"/"position"
# are handled dynamically there instead (one-hot groups, or merged with
# bezirk) -- not listed here.
_FIELD_TO_FEATURE_COLUMNS = {
    "area_m2": ["area_m2"],
    "condition": ["condition"],
    "rooms": ["rooms"],
    "floor": ["floor", "is_ground_floor"],
    "total_floors": ["is_top_floor"],
    "energy_class": ["energy_class"],
    "has_lift": ["has_lift"],
    "has_balcony": ["has_balcony"],
    "furnished": ["furnished"],
    "transit_distance_min": ["transit_distance_min"],
}


def _fix_xgboost_base_score_for_shap(booster) -> None:
    """Patches `booster.save_raw` so shap.TreeExplainer can read this model.

    See MLlogic-sales/predict.py's identical helper for the full
    explanation -- duplicated here rather than shared, matching this
    project's pattern of keeping each model folder self-contained.
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
        if raw[pos:pos + 1] != b"S":
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


def load_explainer(predictor: "RentalPricePredictor"):
    """Builds a SHAP TreeExplainer for `predictor`'s model, for use with
    RentalPricePredictor.explain() above. Construction takes a moment --
    build once, e.g. at API startup, and reuse it across requests.

    Requires the optional `shap` dependency -- imported lazily here so the
    rest of this module works fine without it.
    """
    import shap

    _fix_xgboost_base_score_for_shap(predictor.model.get_booster())
    return shap.TreeExplainer(predictor.model)


# Load model only once when needed
_predictor = None


def predict_rent(property_data):
    """
    Main function for API / AI agent

    Args:
        property_data (dict): Property features

    Returns:
        float: Predicted monthly warm rent (€)
    """

    global _predictor

    if _predictor is None:
        _predictor = RentalPricePredictor()

    result = _predictor.predict(property_data)
    return result["predicted_rent_eur"]
