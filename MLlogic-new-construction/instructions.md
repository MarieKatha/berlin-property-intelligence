# MLlogic-sales

Trains the production XGBoost model that predicts Berlin secondary-sales prices,
and exports it as a single `.pkl` file ready to be loaded by an API. This is the
deployment counterpart to the experimentation done in
`notebooks/notebook_fabian_refined.ipynb` — that notebook used an 80/20
train/test split to compare Decision Tree / Random Forest / XGBoost and tune
hyperparameters; this folder takes the winning configuration (tuned XGBoost)
and refits it on 100% of the available data, since a deployed model should use
every row it can, not hold 20% back for evaluation.

## Files

| File | What it does |
|---|---|
| `config.py` | Paths, ordinal encoding maps, and the tuned XGBoost hyperparameters (copied from the notebook's `RandomizedSearchCV` result) |
| `preprocessing.py` | Feature engineering: log-transforms the target, derives floor/position features, ordinal-encodes `energy_class`/`condition`, one-hot encodes `bezirk`/`transit_line`, and leakage-safe target-encodes `ortsteil` |
| `train.py` | Loads `raw_data/secondary_sales.csv`, builds features, trains the model on all 50,000 rows, and saves `model.pkl` |
| `predict.py` | Loads `model.pkl` and turns a single raw listing (a dict) into a predicted price in EUR — the function an API would import |

## How to run

From this directory (`MLlogic-sales/`), using the project's `berlin-property`
environment (see the repo's `requirements.txt` — `pandas`, `scikit-learn`,
`xgboost`, `joblib` are already listed there, nothing extra to install):

```bash
python train.py
```

This prints something like:

```
Trained on 50000 rows, 35 features
Model bundle saved to .../MLlogic-sales/model.pkl
```

`model.pkl` is **not committed to git** (see the repo's `.gitignore` — `*.pkl` is
already excluded) — re-run `train.py` to (re)generate it locally or in CI/CD
before deploying.

To sanity-check the trained model on one example listing:

```bash
python predict.py
```

## What's inside `model.pkl`

`train.py` saves a single dict via `joblib.dump`, not just the raw model —
everything an API needs to go from a raw listing to a prediction lives in one
file:

```python
{
    "model": <fitted XGBRegressor>,
    "feature_columns": [...],       # exact column order the model expects
    "ortsteil_lookup": <pd.Series>, # neighbourhood -> typical price level
    "ortsteil_global_mean": <float>,# fallback for a neighbourhood never seen in training
}
```

## Using it in an API

```python
from predict import load_model_bundle, predict_price_eur

# Load once at API startup, not per-request
bundle = load_model_bundle()

listing = {
    "ortsteil": "Kreuzberg",
    "bezirk": "Friedrichshain-Kreuzberg",
    "rooms": 2,
    "area_m2": 69.0,
    "floor": 1,
    "total_floors": 6,
    "energy_class": "B",              # A_plus, A, B, C, D, E, F, G, H
    "condition": "saniert",           # renovierungsbeduerftig, renoviert, modernisiert, saniert, kernsaniert
    "has_lift": True,
    "has_balcony": True,
    "has_cellar": False,
    "has_parking": False,
    "transit_line": "U1",
    "transit_distance_min": 10,
    "mortgage_rate_at_listing": 3.5,
    "position": "hinterhaus",         # hinterhaus, seitenfluegel, vorderhaus
}

predicted_price_eur = predict_price_eur(listing, bundle)
```

Listings with an `ortsteil` value that never appeared in training fall back to
the training set's global average price level automatically (no error raised)
— see `predict.py`'s `_build_feature_row`.

## Retraining

If `raw_data/secondary_sales.csv` is updated, or the encoding/hyperparameters
in `config.py` change, just re-run `python train.py` — it overwrites
`model.pkl` with a freshly trained model. There's no separate "update" step;
training is always a full refit on the current CSV.
