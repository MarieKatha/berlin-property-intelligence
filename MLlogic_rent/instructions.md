# MLlogic-rent

Trains the production XGBoost model that predicts Berlin rental prices,
and exports it as a single `.json` file (plus supporting encoder pickles) ready
to be loaded by an API. This is the deployment counterpart to the experimentation
done in `notebooks/notebook_rental_analysis.ipynb` — that notebook used an 80/20
train/test split to compare models and tune hyperparameters; this folder takes
the winning configuration (tuned XGBoost) and refits it on 100% of the available
data, since a deployed model should use every row it can, not hold 20% back for
evaluation.

## Files

| File | What it does |
|---|---|
| `metadata.json` | Model performance metrics (R², MAE, RMSE) and feature column order |
| `xgboost_rental_model.json` | Trained XGBoost model (language-agnostic, smaller than pickle) |
| `target_encoder.pkl` | Fitted `TargetEncoder` for `ortsteil` (neighbourhood → mean rental price) |
| `energy_encoder.pkl` | Fitted `OrdinalEncoder` for `energy_class` (A+ → 0, H → 8) |
| `condition_encoder.pkl` | Fitted `OrdinalEncoder` for `condition` (renovation state) |
| `rental_price_predictor.py` | Loads all artifacts and exposes `predict_rent(property_data)` for API |
| `requirements.txt` | Dependencies: `xgboost`, `pandas`, `category-encoders`, `scikit-learn` |

## How to run

From this directory (`MLlogic_rent/`), using the project environment:

```bash
# Install dependencies (one-time)
pip install -r requirements.txt

# Retrain model on latest data (if raw_data/berlin_rentals.csv updated)
python train.py
```

**Note:** `*.pkl` and `*.json` files are **not committed to git** (see `.gitignore`)
— regenerate locally or in CI/CD before deploying.

To sanity-check on one example listing:

```bash
python -c "from rental_price_predictor import predict_rent; print(predict_rent({...}))"
```

## What's inside the model artifacts

- **`xgboost_rental_model.json`**: Trained model in XGBoost's JSON format (portable, secure)
- **`*_encoder.pkl`**: Fitted encoders for categorical features (reused at prediction time)
- **`metadata.json`**: Feature column order + performance stats:

```json
{
  "feature_columns": ["ortsteil", "bezirk_*, "lat", "lon", ...],
  "model_performance": {
    "r2_score": 0.8512,
    "mae": 145.50,
    "rmse": 256.30
  }
}
```

## Using it in an API

```python
from rental_price_predictor import predict_rent

# Call once per prediction request (lazy-loads model on first call)
listing = {
    "ortsteil": "Charlottenburg",
    "bezirk": "Charlottenburg-Wilmersdorf",
    "lat": 52.5200,
    "lon": 13.4050,
    "rooms": 3,
    "area_m2": 85,
    "floor": 2,
    "total_floors": 4,
    "year_built": 1980,
    "energy_class": "D",                  # A_plus, A, B, C, D, E, F, G, H
    "condition": "renoviert",             # renovierungsbedürftig, renoviert, saniert, modernisiert
    "has_lift": 0,
    "has_balcony": 1,
    "has_cellar": 0,
    "has_parking": 0,
    "transit_line": "U7",
    "transit_distance_min": 5,
    "mortgage_rate_at_listing": 3.5,
    "position": "vorderhaus",             # hinterhaus, seitenflügel, vorderhaus
    "is_top_floor": 0,
    "is_ground_floor": 0
}

predicted_rent_eur = predict_rent(listing)
# Returns: 1250.50 (€ per month, warm rent)
```

Listings with an `ortsteil` value never seen in training automatically fall back
to the training set's global mean rent (no error raised) — see
`rental_price_predictor.py`'s `preprocess()` method.

## Model Features

**Target:** `warmmiete_eur_monthly` (warm rent including utilities)

**Features:**
- **Target-encoded:** `ortsteil` (neighbourhood)
- **Ordinal-encoded:** `energy_class`, `condition`
- **One-hot-encoded:** `bezirk`, `transit_line`, `position`
- **Binary:** `has_lift`, `has_balcony`, `has_cellar`, `has_parking`, `is_top_floor`, `is_ground_floor`
- **Continuous:** `lat`, `lon`, `rooms`, `area_m2`, `floor`, `year_built`, `transit_distance_min`, `mortgage_rate_at_listing`

## Performance

| Metric | Value |
|--------|-------|
| **R² Score (Test)** | 0.8512 |
| **MAE (€/month)** | €145.50 |
| **RMSE (€/month)** | €256.30 |
| **Training Rows** | 40,000 |
| **Features** | 47 |

## Retraining

If `raw_data/berlin_rentals.csv` is updated, or the hyperparameters
in `train.py` change, simply re-run:

```bash
python train.py
```

This overwrites all model artifacts with a freshly trained model using 100% of
the current data. There's no separate "update" step; training is always a
complete refit.

## Troubleshooting

**Model won't load:**
- Ensure all `.pkl` and `.json` files are in the same directory as `rental_price_predictor.py`
- Check `requirements.txt` packages are installed: `pip install -r requirements.txt`

**"Unauthorized" when scraping IS24 data:**
- The website blocks automated requests. Use Selenium (headless browser) or contact IS24 for API access.

**Predictions seem off:**
- Re-run `train.py` on latest data if dataset has changed
- Check `metadata.json` for model performance stats
