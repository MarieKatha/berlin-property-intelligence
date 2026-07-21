"""FastAPI app exposing the full secondary-sales price model (MLlogic-sales/)
trained on 18 features (location, size, condition, energy class, floor,
amenities, ...).

This file only serves the model -- it doesn't retrain or duplicate any
preprocessing logic. All of that lives in MLlogic-sales/ (config.py,
preprocessing.py, train.py, predict.py); we just import and call it.

Only `ortsteil`, `area_m2` and `condition` are required; every other field
(including `listing_price`) is optional and gets filled with its
training-set average (fill_missing=True is always on here -- see the
accuracy caveat in MLlogic-sales/predict.py's predict_price_eur docstring,
since predictions naturally get less reliable the fewer fields are actually
provided).

`listing_price` isn't a model feature -- it's the listing's actual/asking
price, provided so the response can report how it compares to the model's
prediction (listing_vs_predicted_pct: >100% means priced above the
prediction, <100% means priced below it). If it's omitted, the response
skips that comparison and explains how to get it instead.
"""
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# MLlogic-sales is a plain folder of scripts (its name has a hyphen, so it can't
# be imported as a normal Python package, e.g. `from MLlogic-sales import ...`
# is a syntax error) -- add it to sys.path so its modules import as top-level
# names instead, exactly like train.py/predict.py already do when run directly.
MLLOGIC_SALES_DIR = Path(__file__).resolve().parent.parent / "MLlogic-sales"
sys.path.insert(0, str(MLLOGIC_SALES_DIR))

from predict import load_model_bundle, predict_price_eur  # noqa: E402

app = FastAPI(title="Berlin Secondary Sales Price API (full model)")

try:
    model_bundle = load_model_bundle()
except FileNotFoundError:
    model_bundle = None


# Enums render as dropdowns in the Swagger UI (/docs) instead of free-text boxes.
# Values (the right-hand side) match the raw categories in raw_data/secondary_sales.csv
# exactly -- those are what's actually sent as the query value.


class EnergyClass(str, Enum):
    """Ordinal: A_plus = most efficient, H = least efficient."""

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
    """Ordinal: renovierungsbeduerftig = worst, kernsaniert = best."""

    renovierungsbeduerftig = "renovierungsbedürftig"
    renoviert = "renoviert"
    modernisiert = "modernisiert"
    saniert = "saniert"
    kernsaniert = "kernsaniert"


class Position(str, Enum):
    """One-hot: building position on the plot."""

    gartenhaus = "gartenhaus"
    hinterhaus = "hinterhaus"
    seitenfluegel = "seitenflügel"
    vorderhaus = "vorderhaus"


# ortsteil has 82 distinct values (target-encoded, not one-hot, in the model --
# see MLlogic-sales/preprocessing.py) -- too many to hand-write as a class body
# without risking a typo, so the Enum is built programmatically from the exact
# values in raw_data/secondary_sales.csv instead.
_ORTSTEIL_VALUES = [
    "Adlershof", "Alt-Treptow", "Baumschulenweg", "Biesdorf", "Blankenburg",
    "Borsigwalde", "Britz", "Buch", "Buckow", "Charlottenburg",
    "Charlottenburg-Nord", "Dahlem", "Falkenberg", "Falkenhagener Feld",
    "Französisch Buchholz", "Friedenau", "Friedrichsfelde", "Friedrichshagen",
    "Friedrichshain", "Frohnau", "Gatow", "Gesundbrunnen", "Gropiusstadt",
    "Grunewald", "Grünau", "Halensee", "Hansaviertel", "Haselhorst",
    "Heiligensee", "Hellersdorf", "Hermsdorf", "Hohenschönhausen", "Karlshorst",
    "Karow", "Kaulsdorf", "Kladow", "Konradshöhe", "Kreuzberg",
    "Köpenick (Ort)", "Lankwitz", "Lichtenberg (Ort)", "Lichtenrade",
    "Lichterfelde", "Lübars", "Mahlsdorf", "Malchow", "Mariendorf",
    "Marienfelde", "Marzahn", "Mitte (Ort)", "Moabit", "Märkisches Viertel",
    "Müggelheim", "Neukölln (Ort)", "Niederschönhausen", "Nikolassee",
    "Pankow (Ort)", "Plänterwald", "Prenzlauer Berg", "Reinickendorf (Ort)",
    "Rudow", "Rummelsburg", "Schmargendorf", "Schmöckwitz", "Schöneberg",
    "Siemensstadt", "Spandau (Ort)", "Staaken", "Steglitz", "Tegel",
    "Tempelhof", "Tiergarten", "Treptow", "Waidmannslust", "Wannsee",
    "Wartenberg", "Wedding", "Weißensee", "Westend", "Wilmersdorf",
    "Wittenau", "Zehlendorf",
]


def _enum_member_name(value: str) -> str:
    name = value.lower()
    for umlaut, ascii_form in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        name = name.replace(umlaut, ascii_form)
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


Ortsteil = Enum("Ortsteil", {_enum_member_name(v): v for v in _ORTSTEIL_VALUES}, type=str)
Ortsteil.__doc__ = "Berlin neighbourhood (target-encoded by the model, not one-hot)."


class PredictResponse(BaseModel):
    predicted_price_eur: float
    listing_vs_predicted_pct: Optional[float] = None
    message: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    minimal_predict_link = "/predict?ortsteil=Kreuzberg&area_m2=69&condition=saniert"
    minimal_with_listing_price_link = minimal_predict_link + "&listing_price=600000"
    full_predict_link = (
        "/predict?ortsteil=Kreuzberg&area_m2=69&condition=saniert&listing_price=600000"
        "&rooms=2"
        "&floor=1&total_floors=6&year_built=2011&energy_class=B"
        "&has_lift=true&has_balcony=true&has_cellar=false&has_parking=false"
        "&transit_distance_min=10&mortgage_rate_at_listing=3.5&position=hinterhaus"
    )
    docs_link = "/docs"
    return (
        "Welcome to the Berlin Secondary Sales Price API (full model).<br>"
        f'Minimal prediction, no listing_price (just the required fields): <a href="{minimal_predict_link}">{minimal_predict_link}</a><br>'
        f'Minimal prediction with listing_price: <a href="{minimal_with_listing_price_link}">{minimal_with_listing_price_link}</a><br>'
        f'Full-listing prediction (every field known): <a href="{full_predict_link}">{full_predict_link}</a><br>'
        f'For the interactive API docs, follow this link: <a href="{docs_link}">{docs_link}</a>'
    )


@app.get("/predict", response_model=PredictResponse)
def predict(
    ortsteil: Ortsteil = Query(..., description="Berlin neighbourhood (required)"),
    area_m2: float = Query(..., gt=0, description="Living area in square meters (required)"),
    condition: Condition = Query(..., description="Property condition (required)"),
    listing_price: Optional[float] = Query(
        None,
        gt=0,
        description="The listing's actual/asking price in EUR. Optional -- if given, the "
        "response compares it against the model's prediction.",
    ),
    rooms: Optional[int] = Query(None, gt=0),
    floor: Optional[int] = Query(None, ge=0),
    total_floors: Optional[int] = Query(None, ge=0),
    year_built: Optional[int] = Query(None, ge=1800, le=2100),
    energy_class: Optional[EnergyClass] = Query(None),
    has_lift: Optional[bool] = Query(None),
    has_balcony: Optional[bool] = Query(None),
    has_cellar: Optional[bool] = Query(None),
    has_parking: Optional[bool] = Query(None),
    transit_distance_min: Optional[int] = Query(None, ge=0),
    mortgage_rate_at_listing: Optional[float] = Query(None, ge=0),
    position: Optional[Position] = Query(None),
) -> PredictResponse:
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python MLlogic-sales/train.py` first.")

    listing = {
        "ortsteil": ortsteil, "area_m2": area_m2, "condition": condition,
        "rooms": rooms, "floor": floor, "total_floors": total_floors,
        "year_built": year_built, "energy_class": energy_class, "has_lift": has_lift,
        "has_balcony": has_balcony, "has_cellar": has_cellar, "has_parking": has_parking,
        "transit_distance_min": transit_distance_min,
        "mortgage_rate_at_listing": mortgage_rate_at_listing, "position": position,
    }
    # Enum members are (str, Enum) subclasses, but unwrap to plain strings so the
    # listing dict passed downstream only ever contains plain Python types.
    listing = {k: (v.value if isinstance(v, Enum) else v) for k, v in listing.items() if v is not None}

    try:
        # fill_missing is always on: only ortsteil/area_m2/condition are required
        # above, so every other field is routinely absent by design.
        predicted_price_eur = predict_price_eur(listing, model_bundle, fill_missing=True)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if listing_price is None:
        return PredictResponse(
            predicted_price_eur=round(predicted_price_eur, 2),
            message="if you specify the actual listing price I can compare it with my "
            "prediction and rate the value of the listing",
        )

    # factor > 1 -> the listing is priced above the model's prediction (possibly
    # overpriced); factor <= 1 -> priced at or below it (possibly underpriced or
    # a good deal).
    factor = listing_price / predicted_price_eur
    listing_vs_predicted_pct = factor * 100

    if factor > 1:
        message = f"This offer is {round((factor - 1) * 100, 2)}% more expensive than my prediction."
    else:
        message = f"This offer is {round((1 - factor) * 100, 2)}% cheaper than my prediction."

    return PredictResponse(
        predicted_price_eur=round(predicted_price_eur, 2),
        listing_vs_predicted_pct=round(listing_vs_predicted_pct, 2),
        message=message,
    )
