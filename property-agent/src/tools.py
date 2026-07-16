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

@tool(parse_docstring=True)
def generate_password(length: int, include_special_chars: bool) -> str:
    """
    Generates a secure random password with a given length and possibility to
    include special characters. Both parameters are set by the user and must be
    asked

    Args:
        length: number of characters of the password
        include_special_chars: whether password characters contain special chars
            or not
    """
    try:
        import random
        import string
        chars = string.ascii_letters + string.digits
        if include_special_chars:
            chars += string.punctuation
        return "".join(random.choice(chars) for _ in range(length))
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

@tool
def predict_price(area_m2: float) -> str:
    """
    Predicts the sale price in EUR for a given living area in Berlin.

    Args:
        area_m2: Living area in square meters (must be greater than 0)
    """
    api_url = os.getenv("API_URL", "http://localhost:8000")
    response = requests.get(f"{api_url}/predict", params={"area_m2": area_m2})
    result = response.json()
    return f"{result['price_eur']} EUR"

@tool(parse_docstring=True)
def predict_sales_price(
    ortsteil: str,
    bezirk: str,
    rooms: int,
    area_m2: float,
    floor: int,
    total_floors: int,
    energy_class: str,
    condition: str,
    has_lift: bool,
    has_balcony: bool,
    has_cellar: bool,
    has_parking: bool,
    transit_line: str,
    transit_distance_min: int,
    mortgage_rate_at_listing: float,
    position: str,
) -> str:
    """
    Predicts the secondary-sales price in EUR for a full Berlin listing using
    the tuned XGBoost model (MLlogic-sales/predict.py's predict_price_eur).
    Ask the user for any field you don't already know before calling this —
    all of them are required by the model.

    Note: the API endpoint this calls (POST /predict_sales_price) does not
    exist yet on the model's FastAPI app (api/fast.py currently only exposes
    the simple area_m2-only /predict). This tool is wired up ahead of that so
    it's ready once the endpoint is added.

    Args:
        ortsteil: neighbourhood, e.g. "Kreuzberg"
        bezirk: district, e.g. "Friedrichshain-Kreuzberg"
        rooms: number of rooms
        area_m2: living area in square meters
        floor: floor the unit is on (0 = ground floor)
        total_floors: total floors in the building
        energy_class: one of A_plus, A, B, C, D, E, F, G, H (best to worst)
        condition: one of renovierungsbeduerftig, renoviert, modernisiert,
            saniert, kernsaniert (worst to best)
        has_lift: whether the building has a lift
        has_balcony: whether the unit has a balcony
        has_cellar: whether the unit has a cellar
        has_parking: whether a parking spot is included
        transit_line: nearest transit line, e.g. "U1"
        transit_distance_min: walking minutes to that transit stop
        mortgage_rate_at_listing: prevailing mortgage rate in percent, e.g. 3.5
        position: one of vorderhaus, hinterhaus, seitenfluegel
    """
    try:
        api_url = os.getenv("API_URL", "http://localhost:8000")
        listing = {
            "ortsteil": ortsteil,
            "bezirk": bezirk,
            "rooms": rooms,
            "area_m2": area_m2,
            "floor": floor,
            "total_floors": total_floors,
            "energy_class": energy_class,
            "condition": condition,
            "has_lift": has_lift,
            "has_balcony": has_balcony,
            "has_cellar": has_cellar,
            "has_parking": has_parking,
            "transit_line": transit_line,
            "transit_distance_min": transit_distance_min,
            "mortgage_rate_at_listing": mortgage_rate_at_listing,
            "position": position,
        }
        response = requests.post(f"{api_url}/predict_sales_price", json=listing)
        response.raise_for_status()
        result = response.json()
        return f"{result['price_eur']} EUR"
    except Exception as e:
        return f"Tool error: Please check your input and try again. ({e})"
