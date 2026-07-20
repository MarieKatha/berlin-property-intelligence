"""Trains the final secondary-sales price model and exports it for deployment.

Unlike the notebooks (which use an 80/20 train/test split to *evaluate* and
compare models), this script trains on the FULL dataset -- there is no
held-out test set here. This is the production artifact, not a model-selection
experiment: the train/test evaluation in notebooks/notebook_fabian_refined.ipynb
already established that tuned XGBoost is the best model (test MAE ~€28,007,
test R² ~0.974) and picked its hyperparameters (see config.py). Here we just
refit that same configuration on every available row, to give the deployed
model the most data possible.

Output: MLlogic-sales/model.pkl -- a dict (see bottom of this file) containing
everything an API needs to preprocess a new listing and predict its price.
"""
import joblib
from xgboost import XGBRegressor

from config import MODEL_OUTPUT_PATH, RAW_DATA_PATH, TARGET, XGB_PARAMS
from preprocessing import (
    build_raw_features,
    encode_ordinal,
    fit_ortsteil_lookup,
    load_raw_data,
    target_encode_out_of_fold,
)


def train() -> dict:
    df_sales = load_raw_data(RAW_DATA_PATH)
    df = build_raw_features(df_sales)
    df = encode_ordinal(df)

    y = df[TARGET]

    # Leakage-safe out-of-fold encoding of ortsteil for TRAINING: the model never
    # sees a row's own price baked into its own ortsteil feature.
    df["ortsteil_target_enc"] = target_encode_out_of_fold(df["ortsteil"], y)

    # Full-data lookup table for ortsteil, saved alongside the model and reused
    # to encode any listing seen after training (this is the deployed-model
    # analogue of what the notebook used to encode its held-out test set).
    ortsteil_lookup, ortsteil_global_mean = fit_ortsteil_lookup(df["ortsteil"], y)

    X = df.drop(columns=[TARGET, "ortsteil"])

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X, y)

    # Per-feature training-set average, used by predict.py's fill_missing=True path
    # to fill in features a caller couldn't provide (e.g. only ortsteil/area_m2/
    # condition known) -- the same "typical value" imputation validated in the
    # "Robustness check" section of notebooks/notebook_fabian_refined.ipynb.
    feature_fill_values = X.mean()

    bundle = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "feature_fill_values": feature_fill_values,
        "ortsteil_lookup": ortsteil_lookup,
        "ortsteil_global_mean": ortsteil_global_mean,
    }

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_OUTPUT_PATH)

    print(f"Trained on {len(df)} rows, {X.shape[1]} features")
    print(f"Model bundle saved to {MODEL_OUTPUT_PATH}")

    return bundle


if __name__ == "__main__":
    train()
