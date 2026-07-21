from datetime import datetime
from langchain.tools import tool
from geopy.geocoders import Nominatim
import requests
import os

@tool
def get_now():
    """
    Returns current day, date, and time in presented format
    Can be used as a reference to calculate past and future
    moments of time. If user asks only fpr time, date or day,
    only return the parameter the user has asked for
    """
    try:
        return datetime.now().strftime("%A, %d.%m.%Y %H:%M:%S")
    except Exception as e:
        return f"Tool error: Please check your input and try again. ({e})"

@tool
def get_lat_lon_osm(address: str) -> tuple[float | None, float | None]:
    """Get lat/lon from address using OpenStreetMap (Nominatim)."""

    geocoder = Nominatim(user_agent="berlin_realestate")

    try:
        location = geocoder.geocode(address)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception:
        return None, None


if __name__ == "__main__":
    lat, lon = get_lat_lon_osm("Alexander Platz")
    print(f"Lat: {lat}, Lon: {lon}")

@tool(parse_docstring=True)
def predict_sales_price(
    ortsteil: str,
    area_m2: float,
    condition: str,
    listing_price: float | None = None,
    rooms: int | None = None,
    floor: int | None = None,
    total_floors: int | None = None,
    year_built: int | None = None,
    energy_class: str | None = None,
    has_lift: bool | None = None,
    has_balcony: bool | None = None,
    has_cellar: bool | None = None,
    has_parking: bool | None = None,
    transit_distance_min: int | None = None,
    mortgage_rate_at_listing: float | None = None,
    position: str | None = None,
) -> str:
    """
    Predicts the secondary-sales price in EUR for a Berlin listing using the
    full XGBoost model served at api/fast.py's /predict_sales (backed by
    MLlogic-sales). Only ortsteil, area_m2 and condition are required to call
    this tool -- every other field is optional and, when provided, improves
    prediction accuracy. How to collect these fields from the user is
    governed by the system prompt, not by this description. If listing_price
    is given, the response also rates how the listing compares to the
    prediction.

    Args:
        ortsteil: Berlin neighbourhood, e.g. "Kreuzberg", "Prenzlauer Berg"
            (required)
        area_m2: living area in square meters (required)
        condition: one of renovierungsbedürftig, renoviert, modernisiert,
            saniert, kernsaniert (worst to best) (required)
        listing_price: the listing's actual/asking price in EUR, if known --
            enables a comparison against the model's prediction
        rooms: number of rooms
        floor: floor the unit is on (0 = ground floor)
        total_floors: total floors in the building
        year_built: construction year
        energy_class: one of A_plus, A, B, C, D, E, F, G, H (best to worst)
        has_lift: whether the building has a lift
        has_balcony: whether the unit has a balcony
        has_cellar: whether the unit has a cellar
        has_parking: whether a parking spot is included
        transit_distance_min: walking minutes to the nearest transit stop
        mortgage_rate_at_listing: prevailing mortgage rate in percent, e.g. 3.5
        position: one of gartenhaus, hinterhaus, seitenflügel, vorderhaus
    """
    try:
        api_url = os.getenv("API_SALES_URL", "http://localhost:8001")
        params = {
            "ortsteil": ortsteil,
            "area_m2": area_m2,
            "condition": condition,
            "listing_price": listing_price,
            "rooms": rooms,
            "floor": floor,
            "total_floors": total_floors,
            "year_built": year_built,
            "energy_class": energy_class,
            "has_lift": has_lift,
            "has_balcony": has_balcony,
            "has_cellar": has_cellar,
            "has_parking": has_parking,
            "transit_distance_min": transit_distance_min,
            "mortgage_rate_at_listing": mortgage_rate_at_listing,
            "position": position,
        }
        params = {k: v for k, v in params.items() if v is not None}
        response = requests.get(f"{api_url}/predict_sales", params=params)
        response.raise_for_status()
        result = response.json()
        reply = f"{result['predicted_price_eur']} EUR"
        if result.get("message"):
            reply += f" ({result['message']})"
        return reply
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except ValueError:
            detail = e.response.text
        return (
            f"Tool error: the API rejected this input ({detail}). Pick a "
            "value from the allowed list above and try again -- don't ask "
            "the user unless none of the allowed values plausibly match."
        )
    except Exception as e:
        return f"Tool error: Please check your input and try again. ({e})"
