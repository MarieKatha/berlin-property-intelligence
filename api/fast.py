"""FastAPI app exposing all three Berlin property price models behind one
service:

  /predict_sales         -- secondary-sales price (MLlogic-sales/)
  /predict_construction  -- new-construction price (MLlogic-new-construction/)
  /predict_rentals       -- monthly warm rent (MLlogic_rent/)

This file only serves the models -- it doesn't retrain or duplicate any
preprocessing logic, each endpoint just imports and calls the relevant
folder's existing prediction code.

MLlogic-sales/ and MLlogic-new-construction/ both have hyphens in their
folder names, so neither can be imported the normal way (`from MLlogic-sales
import ...` is a syntax error), and both happen to define same-named
`config.py`/`preprocessing.py`/`predict.py` modules -- naively sys.path-
inserting both folders would make the second one silently shadow the
first's modules. `_load_predict_module` below loads each folder's
`predict.py` in isolation to avoid that collision (see its docstring).
MLlogic_rent/ has no hyphen and is already a proper package (it has an
`__init__.py`), so it's imported normally.
"""
import importlib
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from MLlogic_rent.rental_price_predictor import RentalPricePredictor  # noqa: E402


def _load_predict_module(folder: Path):
    """Imports <folder>/predict.py as an isolated module object, without
    leaking its `config`/`preprocessing` helper imports into sys.modules
    under their generic names -- otherwise, whichever of MLlogic-sales/ or
    MLlogic-new-construction/ is loaded second would import the first
    folder's config.py/preprocessing.py instead of its own (both define
    modules with those same names).
    """
    generic_names = ("config", "preprocessing", "predict")
    stashed = {name: sys.modules.pop(name, None) for name in generic_names}
    sys.path.insert(0, str(folder))
    try:
        module = importlib.import_module("predict")
    finally:
        sys.path.remove(str(folder))
        for name in generic_names:
            sys.modules.pop(name, None)
        for name, mod in stashed.items():
            if mod is not None:
                sys.modules[name] = mod
    return module


sales_predict = _load_predict_module(REPO_ROOT / "MLlogic-sales")
construction_predict = _load_predict_module(REPO_ROOT / "MLlogic-new-construction")

app = FastAPI(title="Berlin Property Intelligence API")

try:
    sales_bundle = sales_predict.load_model_bundle()
except FileNotFoundError:
    sales_bundle = None

try:
    construction_bundle = construction_predict.load_model_bundle()
except FileNotFoundError:
    construction_bundle = None

try:
    rental_predictor = RentalPricePredictor()
except Exception:
    rental_predictor = None


# Enums render as dropdowns in the Swagger UI (/docs) instead of free-text boxes.
# Values (the right-hand side) match the raw categories in raw_data/*.csv
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
    """Ordinal: renovierungsbeduerftig = worst, kernsaniert = best. Used by
    predict_sales -- predict_rentals uses RentalCondition instead (see its
    docstring for why).
    """

    renovierungsbeduerftig = "renovierungsbedürftig"
    renoviert = "renoviert"
    modernisiert = "modernisiert"
    saniert = "saniert"
    kernsaniert = "kernsaniert"


class RentalCondition(str, Enum):
    """Same idea as Condition, but without kernsaniert: the rental model's
    condition encoder (MLlogic_rent/condition_encoder.pkl) was fit without
    that category ever appearing in its training data, so passing it would
    silently encode as "unknown" rather than raise -- excluded here so the
    dropdown can't offer a value that quietly corrupts the prediction.
    """

    renovierungsbeduerftig = "renovierungsbedürftig"
    renoviert = "renoviert"
    modernisiert = "modernisiert"
    saniert = "saniert"


class Position(str, Enum):
    """One-hot: building position on the plot. Not a feature of the
    new-construction model -- only predict_sales and predict_rentals use it.
    """

    gartenhaus = "gartenhaus"
    hinterhaus = "hinterhaus"
    seitenfluegel = "seitenflügel"
    vorderhaus = "vorderhaus"


class TransitLine(str, Enum):
    """One-hot: nearest transit line. Not a feature of the sales model
    (deliberately dropped there, see MLlogic-sales/preprocessing.py) --
    only predict_construction and predict_rentals use it.
    """

    s_ringbahn = "S Ringbahn"
    s1 = "S1"
    stadtbahn = "Stadtbahn"
    u1 = "U1"
    u2 = "U2"
    u7 = "U7"
    u8 = "U8"


# ortsteil has 82 distinct values (target-encoded, not one-hot, in every model
# here) -- too many to hand-write as a class body without risking a typo, so
# the Enum is built programmatically from the exact values in
# raw_data/secondary_sales.csv instead. It's a superset of what appears in
# raw_data/new_construction.csv / raw_data/rentals.csv (both same city, just
# fewer ortsteil actually have new-construction listings) -- any of the 82
# values is valid input for every endpoint below.
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
Ortsteil.__doc__ = "Berlin neighbourhood (target-encoded by every model here, not one-hot)."


# predict_construction and predict_rentals both one-hot encode bezirk, but
# neither exposes it as a request parameter -- every ortsteil maps to exactly
# one bezirk (verified against all three raw_data/*.csv files), so it's
# derived automatically instead of asking the caller to keep it in sync with
# ortsteil themselves. (predict_sales doesn't use bezirk at all -- see its
# docstring below.)
ORTSTEIL_TO_BEZIRK = {
    "Adlershof": "Treptow-Köpenick", "Alt-Treptow": "Treptow-Köpenick",
    "Baumschulenweg": "Treptow-Köpenick", "Biesdorf": "Marzahn-Hellersdorf",
    "Blankenburg": "Pankow", "Borsigwalde": "Reinickendorf",
    "Britz": "Neukölln", "Buch": "Pankow", "Buckow": "Neukölln",
    "Charlottenburg": "Charlottenburg-Wilmersdorf",
    "Charlottenburg-Nord": "Charlottenburg-Wilmersdorf",
    "Dahlem": "Steglitz-Zehlendorf", "Falkenberg": "Lichtenberg",
    "Falkenhagener Feld": "Spandau", "Französisch Buchholz": "Pankow",
    "Friedenau": "Tempelhof-Schöneberg", "Friedrichsfelde": "Lichtenberg",
    "Friedrichshagen": "Treptow-Köpenick",
    "Friedrichshain": "Friedrichshain-Kreuzberg", "Frohnau": "Reinickendorf",
    "Gatow": "Spandau", "Gesundbrunnen": "Mitte", "Gropiusstadt": "Neukölln",
    "Grunewald": "Charlottenburg-Wilmersdorf", "Grünau": "Treptow-Köpenick",
    "Halensee": "Charlottenburg-Wilmersdorf", "Hansaviertel": "Mitte",
    "Haselhorst": "Spandau", "Heiligensee": "Reinickendorf",
    "Hellersdorf": "Marzahn-Hellersdorf", "Hermsdorf": "Reinickendorf",
    "Hohenschönhausen": "Lichtenberg", "Karlshorst": "Lichtenberg",
    "Karow": "Pankow", "Kaulsdorf": "Marzahn-Hellersdorf",
    "Kladow": "Spandau", "Konradshöhe": "Reinickendorf",
    "Kreuzberg": "Friedrichshain-Kreuzberg",
    "Köpenick (Ort)": "Treptow-Köpenick", "Lankwitz": "Steglitz-Zehlendorf",
    "Lichtenberg (Ort)": "Lichtenberg", "Lichtenrade": "Tempelhof-Schöneberg",
    "Lichterfelde": "Steglitz-Zehlendorf", "Lübars": "Reinickendorf",
    "Mahlsdorf": "Marzahn-Hellersdorf", "Malchow": "Lichtenberg",
    "Mariendorf": "Tempelhof-Schöneberg",
    "Marienfelde": "Tempelhof-Schöneberg", "Marzahn": "Marzahn-Hellersdorf",
    "Mitte (Ort)": "Mitte", "Moabit": "Mitte",
    "Märkisches Viertel": "Reinickendorf", "Müggelheim": "Treptow-Köpenick",
    "Neukölln (Ort)": "Neukölln", "Niederschönhausen": "Pankow",
    "Nikolassee": "Steglitz-Zehlendorf", "Pankow (Ort)": "Pankow",
    "Plänterwald": "Treptow-Köpenick", "Prenzlauer Berg": "Pankow",
    "Reinickendorf (Ort)": "Reinickendorf", "Rudow": "Neukölln",
    "Rummelsburg": "Lichtenberg", "Schmargendorf": "Charlottenburg-Wilmersdorf",
    "Schmöckwitz": "Treptow-Köpenick", "Schöneberg": "Tempelhof-Schöneberg",
    "Siemensstadt": "Spandau", "Spandau (Ort)": "Spandau",
    "Staaken": "Spandau", "Steglitz": "Steglitz-Zehlendorf",
    "Tegel": "Reinickendorf", "Tempelhof": "Tempelhof-Schöneberg",
    "Tiergarten": "Mitte", "Treptow": "Treptow-Köpenick",
    "Waidmannslust": "Reinickendorf", "Wannsee": "Steglitz-Zehlendorf",
    "Wartenberg": "Lichtenberg", "Wedding": "Mitte", "Weißensee": "Pankow",
    "Westend": "Charlottenburg-Wilmersdorf",
    "Wilmersdorf": "Charlottenburg-Wilmersdorf", "Wittenau": "Reinickendorf",
    "Zehlendorf": "Steglitz-Zehlendorf",
}
assert set(ORTSTEIL_TO_BEZIRK) == set(_ORTSTEIL_VALUES), "ORTSTEIL_TO_BEZIRK is out of sync with _ORTSTEIL_VALUES"


class PredictResponse(BaseModel):
    predicted_price_eur: float
    listing_vs_predicted_pct: Optional[float] = None
    message: Optional[str] = None


class ConstructionPredictResponse(BaseModel):
    predicted_price_eur: float


class RentPredictResponse(BaseModel):
    predicted_rent_eur: float
    listing_vs_predicted_pct: Optional[float] = None
    message: Optional[str] = None


def _comparison(actual: float, predicted: float) -> tuple[float, str]:
    """Shared by predict_sales and predict_rentals: how the listing's actual
    price/rent compares to the model's prediction, as a percentage plus a
    human-readable message.
    """
    factor = actual / predicted
    if factor > 1:
        message = f"This offer is {round((factor - 1) * 100, 2)}% more expensive than my prediction."
    else:
        message = f"This offer is {round((1 - factor) * 100, 2)}% cheaper than my prediction."
    return round(factor * 100, 2), message


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    sales_link = "/predict_sales?ortsteil=Kreuzberg&area_m2=69&condition=saniert&listing_price=600000"
    construction_link = "/predict_construction?ortsteil=Mitte%20(Ort)&area_m2=65"
    rentals_link = "/predict_rentals?ortsteil=Kreuzberg&area_m2=65&condition=saniert&listing_rent_eur=1200"
    docs_link = "/docs"
    return (
        "Welcome to the Berlin Property Intelligence API.<br>"
        f'Secondary-sales price prediction: <a href="{sales_link}">{sales_link}</a><br>'
        f'New-construction price prediction: <a href="{construction_link}">{construction_link}</a><br>'
        f'Rental price prediction: <a href="{rentals_link}">{rentals_link}</a><br>'
        f'For the interactive API docs, follow this link: <a href="{docs_link}">{docs_link}</a>'
    )


@app.get("/predict_sales", response_model=PredictResponse)
def predict_sales(
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
    """Predicts a secondary-sales price with the 18-feature model trained in
    MLlogic-sales/ (location, size, condition, energy class, floor,
    amenities, ...).

    Only `ortsteil`, `area_m2` and `condition` are required; every other
    field (including `listing_price`) is optional and gets filled with its
    training-set average (fill_missing=True is always on here -- see the
    accuracy caveat in MLlogic-sales/predict.py's predict_price_eur
    docstring, since predictions naturally get less reliable the fewer
    fields are actually provided).

    `bezirk` and `transit_line` are NOT request parameters here: neither is
    a model feature (see MLlogic-sales/preprocessing.py -- ortsteil's target
    encoding already captures location more precisely than bezirk's coarser
    12-district grouping, and transit_line simply wasn't found useful enough
    to keep).
    """
    if sales_bundle is None:
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
        predicted_price_eur = sales_predict.predict_price_eur(listing, sales_bundle, fill_missing=True)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if listing_price is None:
        return PredictResponse(
            predicted_price_eur=round(predicted_price_eur, 2),
            message="if you specify the actual listing price I can compare it with my "
            "prediction and rate the value of the listing",
        )

    listing_vs_predicted_pct, message = _comparison(listing_price, predicted_price_eur)
    return PredictResponse(
        predicted_price_eur=round(predicted_price_eur, 2),
        listing_vs_predicted_pct=listing_vs_predicted_pct,
        message=message,
    )


@app.get("/predict_construction", response_model=ConstructionPredictResponse)
def predict_construction(
    ortsteil: Ortsteil = Query(..., description="Berlin neighbourhood (required)"),
    area_m2: float = Query(..., gt=0, description="Living area in square meters (required)"),
    rooms: Optional[int] = Query(None, gt=0),
    floor: Optional[int] = Query(None, ge=0),
    total_floors: Optional[int] = Query(None, ge=0),
    completion_year: Optional[int] = Query(None, ge=2020, le=2100),
    energy_class: Optional[EnergyClass] = Query(None),
    has_lift: Optional[bool] = Query(None),
    has_balcony: Optional[bool] = Query(None),
    has_parking: Optional[bool] = Query(None),
    transit_line: Optional[TransitLine] = Query(None),
    transit_distance_min: Optional[int] = Query(None, ge=0),
    mortgage_rate_at_listing: Optional[float] = Query(None, ge=0),
) -> ConstructionPredictResponse:
    """Predicts a new-construction price with the model trained in
    MLlogic-new-construction/.

    Only `ortsteil` and `area_m2` are required; every other field is
    optional and gets filled with its training-set average
    (fill_missing=True is always on here -- see the accuracy caveat in
    MLlogic-new-construction/predict.py's predict_price_eur docstring).
    `bezirk` isn't a parameter: it's derived automatically from `ortsteil`
    (see ORTSTEIL_TO_BEZIRK above). There's no `condition` field --
    new-construction listings don't have one.
    """
    if construction_bundle is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Run `python MLlogic-new-construction/train.py` first."
        )

    listing = {
        "ortsteil": ortsteil, "bezirk": ORTSTEIL_TO_BEZIRK[ortsteil.value],
        "area_m2": area_m2, "rooms": rooms, "floor": floor, "total_floors": total_floors,
        "completion_year": completion_year, "energy_class": energy_class, "has_lift": has_lift,
        "has_balcony": has_balcony, "has_parking": has_parking, "transit_line": transit_line,
        "transit_distance_min": transit_distance_min, "mortgage_rate_at_listing": mortgage_rate_at_listing,
    }
    # Enum members are (str, Enum) subclasses, but unwrap to plain strings so the
    # listing dict passed downstream only ever contains plain Python types.
    listing = {k: (v.value if isinstance(v, Enum) else v) for k, v in listing.items() if v is not None}

    try:
        # fill_missing is always on: only ortsteil/area_m2 are required above,
        # so every other field is routinely absent by design.
        predicted_price_eur = construction_predict.predict_price_eur(listing, construction_bundle, fill_missing=True)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ConstructionPredictResponse(predicted_price_eur=round(predicted_price_eur, 2))


@app.get("/predict_rentals", response_model=RentPredictResponse)
def predict_rentals(
    ortsteil: Ortsteil = Query(..., description="Berlin neighbourhood (required)"),
    area_m2: float = Query(..., gt=0, description="Living area in square meters (required)"),
    condition: RentalCondition = Query(..., description="Property condition (required)"),
    listing_rent_eur: Optional[float] = Query(
        None,
        gt=0,
        description="The listing's actual/asking monthly warm rent in EUR. Optional -- if given, "
        "the response compares it against the model's prediction.",
    ),
    rooms: Optional[int] = Query(None, gt=0),
    floor: Optional[int] = Query(None, ge=0),
    total_floors: Optional[int] = Query(
        None, ge=0, description="Not a model feature by itself -- only used, together with floor, "
        "to derive whether the unit is on the top/ground floor."
    ),
    energy_class: Optional[EnergyClass] = Query(None),
    has_lift: Optional[bool] = Query(None),
    has_balcony: Optional[bool] = Query(None),
    furnished: Optional[bool] = Query(None),
    transit_distance_min: Optional[int] = Query(None, ge=0),
    transit_line: Optional[TransitLine] = Query(None),
    position: Optional[Position] = Query(None),
) -> RentPredictResponse:
    """Predicts a monthly warm rent with the model in MLlogic_rent/.

    Only `ortsteil`, `area_m2` and `condition` are required; everything else
    is optional. `bezirk` isn't a parameter: it's derived automatically from
    `ortsteil` (see ORTSTEIL_TO_BEZIRK above). Note `condition` uses
    RentalCondition, not Condition -- this model's condition encoder doesn't
    recognise `kernsaniert` (see RentalCondition's docstring).
    """
    if rental_predictor is None:
        raise HTTPException(status_code=503, detail="Rental model not loaded.")

    listing = {
        "ortsteil": ortsteil.value,
        "bezirk": ORTSTEIL_TO_BEZIRK[ortsteil.value],
        "area_m2": area_m2,
        "condition": condition.value,
        "rooms": rooms,
        "floor": floor,
        "energy_class": energy_class.value if energy_class else None,
        "has_lift": has_lift,
        "has_balcony": has_balcony,
        "furnished": furnished,
        "transit_distance_min": transit_distance_min,
        "transit_line": transit_line.value if transit_line else None,
        "position": position.value if position else None,
    }
    if floor is not None:
        listing["is_ground_floor"] = floor == 0
        if total_floors is not None:
            listing["is_top_floor"] = floor == total_floors
    listing = {k: v for k, v in listing.items() if v is not None}

    try:
        result = rental_predictor.predict(listing)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    predicted_rent_eur = result["predicted_rent_eur"]

    if listing_rent_eur is None:
        return RentPredictResponse(
            predicted_rent_eur=round(predicted_rent_eur, 2),
            message="if you specify the actual listing rent I can compare it with my "
            "prediction and rate the value of the listing",
        )

    listing_vs_predicted_pct, message = _comparison(listing_rent_eur, predicted_rent_eur)
    return RentPredictResponse(
        predicted_rent_eur=round(predicted_rent_eur, 2),
        listing_vs_predicted_pct=listing_vs_predicted_pct,
        message=message,
    )
