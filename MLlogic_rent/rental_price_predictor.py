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


    def preprocess(self, input_data):
        """
        Convert raw property input into model-ready features

        Args:
            input_data (dict): Property characteristics

        Returns:
            numpy array: Features in correct model order
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
        categorical_features = ["bezirk", "transit_line", "position"]

        for col in categorical_features:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)

        # Match training features. One-hot dummies default to 0 (not-this-category);
        # anything else missing (an ordinal/continuous feature the caller didn't
        # provide) is left as NaN, which this XGBoost model was trained to handle
        # natively (see metadata.json's "missing": NaN) -- more honest than
        # silently guessing 0 for a feature we simply don't know.
        dummy_prefixes = tuple(f"{col}_" for col in categorical_features) + ("building_era_",)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0 if col.startswith(dummy_prefixes) else np.nan

        # Correct feature order
        X = df[self.feature_columns]

        return X.values


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
