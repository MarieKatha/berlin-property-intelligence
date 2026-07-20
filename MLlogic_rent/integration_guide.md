# Rental Price Model Integration

## For the AI Agent

### Using the Rental Model

```python
from MLlogic_rent.client import RentalPriceClient

# Initialize once
rental_client = RentalPriceClient(api_url="http://rental-model-api:8000")

# Use in agent logic
def predict_rent(property_details: dict) -> str:
    """Agent function to predict rent"""

    # Ensure all required fields are present
    required_fields = [
        'ortsteil', 'bezirk', 'lat', 'lon', 'rooms', 'area_m2',
        'floor', 'total_floors', 'year_built', 'energy_class',
        'condition', 'transit_line', 'transit_distance_min',
        'mortgage_rate_at_listing', 'position'
    ]

    # Add defaults for binary features
    property_details.setdefault('has_lift', 0)
    property_details.setdefault('has_balcony', 0)
    property_details.setdefault('has_cellar', 0)
    property_details.setdefault('has_parking', 0)
    property_details.setdefault('is_top_floor', 0)
    property_details.setdefault('is_ground_floor', 0)

    # Predict
    rent = rental_client.predict(property_details)

    if rent:
        return f"Predicted warm rent: €{rent:.2f}/month"
    else:
        return "Could not predict rent (missing data or API error)"
```

### Running with Docker Compose

```bash
# Start all services
docker-compose up

# Verify rental model is running
curl http://localhost:8001/health

# Test from agent
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{...property data...}'
```

### Expected Response

```json
{
  "predicted_rent_eur": 1250.50,
  "model_type": "XGBRegressor",
  "confidence": {
    "r2_score": 0.8512,
    "mae": 145.50,
    "rmse": 256.30
  }
}
```

## Required Input Fields

```python
{
    "ortsteil": str,                    # Neighbourhood
    "bezirk": str,                      # District
    "lat": float,                       # Latitude
    "lon": float,                       # Longitude
    "rooms": int,                       # Number of rooms
    "area_m2": float,                   # Living area
    "floor": int,                       # Floor number
    "total_floors": int,                # Total floors
    "year_built": int,                  # Year built
    "energy_class": str,                # A+, A, B, C, D, E, F, G, H
    "condition": str,                   # renoviert, renovierungsbedürftig, saniert, modernisiert
    "transit_line": str,                # U7, S5, M1, etc
    "transit_distance_min": int,        # Minutes to transit
    "mortgage_rate_at_listing": float,  # Interest rate %
    "position": str,                    # vorderhaus, seitenflügel, hinterhaus
    "has_lift": int,                    # 0 or 1 (optional)
    "has_balcony": int,                 # 0 or 1 (optional)
    "has_cellar": int,                  # 0 or 1 (optional)
    "has_parking": int,                 # 0 or 1 (optional)
    "is_top_floor": int,                # 0 or 1 (optional)
    "is_ground_floor": int              # 0 or 1 (optional)
}
```
