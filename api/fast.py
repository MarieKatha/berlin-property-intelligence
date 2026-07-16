"""FastAPI app exposing a single prediction endpoint for the area_m2 -> price_eur model."""
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "model.pkl"

app = FastAPI(title="Berlin Secondary Sales Price API")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class PredictResponse(BaseModel):
    price_eur: float


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    predict_link = "/predict?area_m2=70"
    docs_link = "/docs"
    return (
        "Welcome to the Berlin Property Intelligence.<br>"
        f'If you want to predict a sale price please follow this link: <a href="{predict_link}">{predict_link}</a><br>'
        f'For the interactive API docs, follow this link: <a href="{docs_link}">{docs_link}</a>'
    )


@tool
def predict_price(area_m2: float) -> str:
    """
    Predicts the sale price in EUR for a given living area in Berlin.

    Args:
        area_m2: Living area in square meters (must be greater than 0)
    """
    api_url = os.getenv("API_URL", "http://localhost:8080")
    response = requests.get(f"{api_url}/predict", params={"area_m2": area_m2})
    result = response.json()
    return f"{result['price_eur']} EUR"
