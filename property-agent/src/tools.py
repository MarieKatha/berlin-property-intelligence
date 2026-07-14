from datetime import datetime
from langchain.tools import tool

@tool
def get_now():
    """
    Returns current day, date, and time in presented format
    Can be used as a reference to calculate past and future
    moments of time
    """
    return datetime.now().strftime("%A, %d.%m.%Y %H:%M:%S")

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
    import random
    import string
    chars = string.ascii_letters + string.digits
    if include_special_chars:
        chars += string.punctuation
    return "".join(random.choice(chars) for _ in range(length))
