
"""
Berlin Rental Price Prediction Model
For use with AI Agent
"""

import pickle
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
import xgboost as xgb

class RentalPricePredictor:
    def __init__(self, model_path='models/xgboost_rental_model.pkl',
                 metadata_path='models/metadata.json',
                 encoders_path='models/'):
        """Load model and encoders"""

        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load encoders
        with open(f'{encoders_path}/target_encoder.pkl', 'rb') as f:
            self.target_encoder = pickle.load(f)

        with open(f'{encoders_path}/energy_encoder.pkl', 'rb') as f:
            self.energy_encoder = pickle.load(f)

        with open(f'{encoders_path}/condition_encoder.pkl', 'rb') as f:
            self.condition_encoder = pickle.load(f)

        self.feature_columns = self.metadata['feature_columns']

    def preprocess(self, input_data):
        """
        Preprocess input data before prediction

        Args:
            input_data: dict with keys matching feature names
                Example:
                {
                    'ortsteil': 'Charlottenburg',
                    'bezirk': 'Charlottenburg-Wilmersdorf',
                    'lat': 52.5200,
                    'lon': 13.4050,
                    'rooms': 3,
                    'area_m2': 85,
                    ...
                }

        Returns:
            X: numpy array ready for prediction
        """

        df = pd.DataFrame([input_data])

        # Encode ortsteil (target encoding)
        df['ortsteil'] = self.target_encoder.transform(df[['ortsteil']])

        # Encode energy_class (ordinal)
        df['energy_class'] = self.energy_encoder.transform(df[['energy_class']])

        # Encode condition (ordinal)
        df['condition'] = self.condition_encoder.transform(df[['condition']])

        # One-hot encode bezirk
        bezirk_dummies = pd.get_dummies(df['bezirk'], prefix='bezirk', drop_first=True)
        df = pd.concat([df, bezirk_dummies], axis=1)
        df = df.drop(columns=['bezirk'])

        # One-hot encode transit_line
        transit_dummies = pd.get_dummies(df['transit_line'], prefix='transit_line', drop_first=True)
        df = pd.concat([df, transit_dummies], axis=1)
        df = df.drop(columns=['transit_line'])

        # One-hot encode position
        position_dummies = pd.get_dummies(df['position'], prefix='position', drop_first=True)
        df = pd.concat([df, position_dummies], axis=1)
        df = df.drop(columns=['position'])

        # Fill missing columns with 0 (if one-hot creates missing cols)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # Select only model features in correct order
        X = df[self.feature_columns].values

        return X

    def predict(self, input_data):
        """
        Make prediction

        Args:
            input_data: dict with rental property features

        Returns:
            dict with prediction and confidence
        """

        X = self.preprocess(input_data)
        prediction = self.model.predict(X)[0]

        return {
            'predicted_rent_eur': float(prediction),
            'model_type': 'XGBRegressor',
            'confidence': {
                'r2_score': self.metadata['model_performance']['r2_score'],
                'mae': self.metadata['model_performance']['mae'],
                'rmse': self.metadata['model_performance']['rmse']
            }
        }

# Initialize predictor
predictor = RentalPricePredictor()

def predict_rent(property_data):
    """
    Simple API function for AI agent

    Args:
        property_data: dict with property features

    Returns:
        float: predicted monthly rent in €
    """
    result = predictor.predict(property_data)
    return result['predicted_rent_eur']
