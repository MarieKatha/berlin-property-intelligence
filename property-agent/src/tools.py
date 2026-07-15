from datetime import datetime
from langchain.tools import tool
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
