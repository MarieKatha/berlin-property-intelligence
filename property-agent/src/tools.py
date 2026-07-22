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


@tool
def scrape_property_listing(url: str) -> str:
    """
    Scrapes a Berlin property listing from an ImmobilienScout24
    (immobilienscout24.de) URL and returns its structured fields (ortsteil,
    area_m2, rooms, condition, energy_class, has_lift, has_balcony,
    furnished, floor, building_era, ...), so they can be used to call
    predict_sales_price, predict_construction_price, or
    predict_rentals_price without the user having to type everything out
    manually. Fields come back as raw scraped text -- e.g. condition might
    be "Renoviert", energy_class "A+", has_lift "Ja"/"Nein" -- translate
    these into the exact values the predict_* tools expect before calling
    them. A field that's missing from the result wasn't found on the page;
    ask the user for it if the relevant predict_* tool requires it.
    """
    try:
        scraper_url = os.getenv("SCRAPER_API_URL", "http://localhost:8003")
        response = requests.get(f"{scraper_url}/scrape", params={"url": url}, timeout=90)
        response.raise_for_status()
        result = response.json()
        # raw_data has the complete set the scraper extracted (condition,
        # floor, building_era, has_balcony, furnished, ...) -- the top-level
        # fields on the response are only a partial subset of it.
        data = result.get("raw_data") or {k: v for k, v in result.items() if k != "raw_data"}
        fields = {k: v for k, v in data.items() if v not in (None, "")}
        return f"Scraped listing at {url}: {fields}"
    except requests.exceptions.Timeout:
        return "Tool error: scraping this listing took too long. Try again in a moment."
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except ValueError:
            detail = e.response.text
        return f"Tool error: could not scrape this listing ({detail})."
    except Exception as e:
        return f"Tool error: Please check your input and try again. ({e})"

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
        response = requests.get(f"{api_url}/predict_sales", params=params, timeout=15)
        response.raise_for_status()
        result = response.json()
        reply = f"{result['predicted_price_eur']} EUR"
        if result.get("message"):
            reply += f" ({result['message']})"
        return reply
    except requests.exceptions.Timeout:
        return "Tool error: the prediction API took too long to respond. Try again in a moment."
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


@tool(parse_docstring=True)
def predict_construction_price(
    ortsteil: str,
    area_m2: float,
    rooms: int | None = None,
    floor: int | None = None,
    total_floors: int | None = None,
    completion_year: int | None = None,
    energy_class: str | None = None,
    has_lift: bool | None = None,
    has_balcony: bool | None = None,
    has_parking: bool | None = None,
    transit_line: str | None = None,
    transit_distance_min: int | None = None,
    mortgage_rate_at_listing: float | None = None,
) -> str:
    """
    Predicts the new-construction price in EUR for a Berlin listing using
    the model served at api/fast.py's /predict_construction (backed by
    MLlogic-new-construction). Only ortsteil and area_m2 are required to
    call this tool -- every other field is optional and, when provided,
    improves prediction accuracy. How to collect these fields from the
    user is governed by the system prompt, not by this description. There
    is no condition field -- new-construction listings don't have one.

    Args:
        ortsteil: Berlin neighbourhood, e.g. "Kreuzberg", "Prenzlauer Berg"
            (required)
        area_m2: living area in square meters (required)
        rooms: number of rooms
        floor: floor the unit is on (0 = ground floor)
        total_floors: total floors in the building
        completion_year: year the building is/was completed
        energy_class: one of A_plus, A, B, C, D, E, F, G, H (best to worst)
        has_lift: whether the building has a lift
        has_balcony: whether the unit has a balcony
        has_parking: whether a parking spot is included
        transit_line: nearest transit line, one of S Ringbahn, S1,
            Stadtbahn, U1, U2, U7, U8
        transit_distance_min: walking minutes to the nearest transit stop
        mortgage_rate_at_listing: prevailing mortgage rate in percent, e.g. 3.5
    """
    try:
        api_url = os.getenv("API_SALES_URL", "http://localhost:8001")
        params = {
            "ortsteil": ortsteil,
            "area_m2": area_m2,
            "rooms": rooms,
            "floor": floor,
            "total_floors": total_floors,
            "completion_year": completion_year,
            "energy_class": energy_class,
            "has_lift": has_lift,
            "has_balcony": has_balcony,
            "has_parking": has_parking,
            "transit_line": transit_line,
            "transit_distance_min": transit_distance_min,
            "mortgage_rate_at_listing": mortgage_rate_at_listing,
        }
        params = {k: v for k, v in params.items() if v is not None}
        response = requests.get(f"{api_url}/predict_construction", params=params, timeout=15)
        response.raise_for_status()
        result = response.json()
        return f"{result['predicted_price_eur']} EUR"
    except requests.exceptions.Timeout:
        return "Tool error: the prediction API took too long to respond. Try again in a moment."
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


@tool(parse_docstring=True)
def predict_rentals_price(
    ortsteil: str,
    area_m2: float,
    condition: str,
    listing_rent_eur: float | None = None,
    rooms: int | None = None,
    floor: int | None = None,
    total_floors: int | None = None,
    energy_class: str | None = None,
    has_lift: bool | None = None,
    has_balcony: bool | None = None,
    furnished: bool | None = None,
    transit_distance_min: int | None = None,
    transit_line: str | None = None,
    position: str | None = None,
) -> str:
    """
    Predicts the monthly warm rent in EUR for a Berlin listing using the
    model served at api/fast.py's /predict_rentals (backed by
    MLlogic_rent). Only ortsteil, area_m2 and condition are required to
    call this tool -- every other field is optional and, when provided,
    improves prediction accuracy. How to collect these fields from the
    user is governed by the system prompt, not by this description. If
    listing_rent_eur is given, the response also rates how the listing
    compares to the prediction.

    Args:
        ortsteil: Berlin neighbourhood, e.g. "Kreuzberg", "Prenzlauer Berg"
            (required)
        area_m2: living area in square meters (required)
        condition: one of renovierungsbedürftig, renoviert, modernisiert,
            saniert (worst to best -- unlike predict_sales_price, kernsaniert
            is not a valid value for rentals) (required)
        listing_rent_eur: the listing's actual/asking monthly warm rent in
            EUR, if known -- enables a comparison against the model's
            prediction
        rooms: number of rooms
        floor: floor the unit is on (0 = ground floor)
        total_floors: total floors in the building
        energy_class: one of A_plus, A, B, C, D, E, F, G, H (best to worst)
        has_lift: whether the building has a lift
        has_balcony: whether the unit has a balcony
        furnished: whether the unit is furnished
        transit_distance_min: walking minutes to the nearest transit stop
        transit_line: nearest transit line, one of S Ringbahn, S1,
            Stadtbahn, U1, U2, U7, U8
        position: one of gartenhaus, hinterhaus, seitenflügel, vorderhaus
    """
    try:
        api_url = os.getenv("API_SALES_URL", "http://localhost:8001")
        params = {
            "ortsteil": ortsteil,
            "area_m2": area_m2,
            "condition": condition,
            "listing_rent_eur": listing_rent_eur,
            "rooms": rooms,
            "floor": floor,
            "total_floors": total_floors,
            "energy_class": energy_class,
            "has_lift": has_lift,
            "has_balcony": has_balcony,
            "furnished": furnished,
            "transit_distance_min": transit_distance_min,
            "transit_line": transit_line,
            "position": position,
        }
        params = {k: v for k, v in params.items() if v is not None}
        response = requests.get(f"{api_url}/predict_rentals", params=params, timeout=15)
        response.raise_for_status()
        result = response.json()
        reply = f"{result['predicted_rent_eur']} EUR"
        if result.get("message"):
            reply += f" ({result['message']})"
        return reply
    except requests.exceptions.Timeout:
        return "Tool error: the prediction API took too long to respond. Try again in a moment."
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
