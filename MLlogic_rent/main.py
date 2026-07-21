"""
FastAPI app exposing the Berlin rental price model (MLlogic_rent/).

Only ortsteil, area_m2 and condition are required.
All other features are optional and filled with defaults.

If actual_rent_eur is provided, the API compares the listing
price against the predicted rent.
"""

import re
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from rental_price_predictor import RentalPricePredictor


app = FastAPI(
    title="Berlin Rental Price API",
    description="Predicts monthly warm rent for Berlin apartments",
    version="1.0"
)


# Load model once
try:
    predictor = RentalPricePredictor()
except Exception as e:
    predictor = None
    print(f"Model loading failed: {e}")


# ============================================================
# Dropdown values for Swagger
# ============================================================

class EnergyClass(str, Enum):
    A_plus = "A_plus"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"


class Condition(str, Enum):
    renovierungsbeduerftig = "renovierungsbedürftig"
    renoviert = "renoviert"
    modernisiert = "modernisiert"
    saniert = "saniert"
    kernsaniert = "kernsaniert"


class Position(str, Enum):
    gartenhaus = "gartenhaus"
    hinterhaus = "hinterhaus"
    seitenfluegel = "seitenflügel"
    vorderhaus = "vorderhaus"


# ============================================================
# Response
# ============================================================

class PredictResponse(dict):
    pass


# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root():

    return """
    Welcome to the Berlin Rental Price API.<br><br>

    Required fields:<br>
    - ortsteil<br>
    - area_m2<br>
    - condition<br><br>

    Open Swagger documentation:<br>
    <a href="/docs">/docs</a>
    """


@app.get("/predict")
def predict(
    ortsteil: str = Query(..., description="Berlin neighbourhood"),
    area_m2: float = Query(..., gt=0),
    condition: Condition = Query(...),

    # Optional
    rooms: Optional[int] = Query(None),
    floor: Optional[int] = Query(None),

    energy_class: Optional[EnergyClass] = Query(None),

    has_lift: Optional[bool] = Query(None),
    has_balcony: Optional[bool] = Query(None),
    furnished: Optional[bool] = Query(None),

    transit_distance_min: Optional[int] = Query(None),
    transit_line: Optional[str] = Query(None),

    position: Optional[Position] = Query(None),

    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),

    is_top_floor: Optional[bool] = Query(None),
    is_ground_floor: Optional[bool] = Query(None),

    actual_rent_eur: Optional[float] = Query(
        None,
        description="Actual monthly rent to compare against prediction"
    )

):

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Rental model not loaded"
        )


    # Build input dictionary

    listing = {
        "ortsteil": ortsteil,
        "area_m2": area_m2,
        "condition": condition.value,

        "rooms": rooms,
        "floor": floor,

        "energy_class": (
            energy_class.value
            if energy_class else None
        ),

        "has_lift": has_lift,
        "has_balcony": has_balcony,
        "furnished": furnished,

        "transit_distance_min": transit_distance_min,
        "transit_line": transit_line,

        "position": (
            position.value
            if position else None
        ),

        "lat": lat,
        "lon": lon,

        "is_top_floor": is_top_floor,
        "is_ground_floor": is_ground_floor,
    }


    # Remove missing fields
    listing = {
        k:v for k,v in listing.items()
        if v is not None
    }


    try:
        result = predictor.predict(listing)

    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )


    predicted = result["predicted_rent_eur"]


    response = {
        "predicted_rent_eur": round(predicted,2),
        "model_type": result["model_type"],
        "confidence": result["confidence"]
    }


    # Compare with actual listing rent

    if actual_rent_eur:

        factor = actual_rent_eur / predicted

        response["market_comparison"] = {

            "actual_rent_eur": actual_rent_eur,

            "factor": round(factor,2),

            "assessment":
                (
                    "above prediction"
                    if factor > 1.05
                    else "below prediction"
                    if factor < 0.95
                    else "close to prediction"
                )
        }


    return response
