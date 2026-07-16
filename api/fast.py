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


@app.get("/predict", response_model=PredictResponse)
def predict(area_m2: float = Query(gt=0, description="Living area in square meters")) -> PredictResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python api/train.py` first.")

    price_eur = model.predict([[area_m2]])[0]
    return PredictResponse(price_eur=round(float(price_eur), 2))
