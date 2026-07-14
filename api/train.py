"""Trains the area_m2 -> price_eur linear regression and saves it to api/model.pkl."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = Path(__file__).parent.parent / "raw_data" / "secondary_sales.csv"
MODEL_PATH = Path(__file__).parent / "model.pkl"


def train():
    df = pd.read_csv(DATA_PATH)
    X = df[["area_m2"]]
    y = df["price_eur"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
