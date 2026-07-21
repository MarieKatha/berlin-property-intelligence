"""
Simple client for the Rental Price Prediction Model
For use by the AI Agent
"""

import requests
from typing import Dict, Optional

class RentalPriceClient:
    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url

    def predict(self, property_data: Dict) -> Optional[float]:
        """
        Predict rental price

        Args:
            property_data: dict with property features

        Returns:
            float: Predicted rent in EUR, or None if error
        """
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=property_data,
                timeout=5
            )
            response.raise_for_status()
            result = response.json()
            return result["predicted_rent_eur"]

        except Exception as e:
            print(f"Error: {e}")
            return None

    def health_check(self) -> bool:
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False


# For quick testing
if __name__ == "__main__":
    client = RentalPriceClient()

    if not client.health_check():
        print("⚠️  API not running")
    else:
        print("✓ API is healthy")

        test_property = {
            "ortsteil": "Charlottenburg",
            "bezirk": "Charlottenburg-Wilmersdorf",
            "lat": 52.52,
            "lon": 13.40,
            "rooms": 3,
            "area_m2": 85,
            "floor": 2,
            "total_floors": 4,
            "year_built": 1980,
            "energy_class": "D",
            "condition": "renoviert",
            "has_lift": 0,
            "has_balcony": 1,
            "has_cellar": 0,
            "has_parking": 0,
            "transit_line": "U7",
            "transit_distance_min": 5,
            "mortgage_rate_at_listing": 3.5,
            "position": "vorderhaus",
            "is_top_floor": 0,
            "is_ground_floor": 0
        }

        rent = client.predict(test_property)
        print(f"Predicted rent: €{rent:.2f}" if rent else "Prediction failed")
