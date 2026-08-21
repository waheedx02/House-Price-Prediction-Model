from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from src.preprocessing import create_features, FEATURE_ORDER


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "house_price_model.cbm"
)

# Load model once when this module is imported.
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

def predict_price(features: dict | pd.DataFrame) -> float:
    """
    Predict house price from raw house features.

    The input is passed through the same preprocessing pipeline
    used during model development.
    """

    # ---------------------------------------------------------
    # 1. Convert input to DataFrame
    # ---------------------------------------------------------
    if isinstance(features, dict):
        data = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        data = features.copy()
    else:
        raise TypeError(
            "features must be either a dictionary or pandas DataFrame."
        )

    # ---------------------------------------------------------
    # 2. Feature engineering
    # ---------------------------------------------------------
    data = create_features(data)

    # ---------------------------------------------------------
    # 3. Validate feature names
    # ---------------------------------------------------------
    actual_features = list(data.columns)

    if actual_features != FEATURE_ORDER:
        missing = [
            feature
            for feature in FEATURE_ORDER
            if feature not in actual_features
        ]

        unexpected = [
            feature
            for feature in actual_features
            if feature not in FEATURE_ORDER
        ]

        raise ValueError(
            "Feature mismatch.\n"
            f"Missing features: {missing}\n"
            f"Unexpected features: {unexpected}\n"
            f"Expected order: {FEATURE_ORDER}\n"
            f"Actual order: {actual_features}"
        )

    # ---------------------------------------------------------
    # 4. Make sure feature order is correct
    # ---------------------------------------------------------
    data = data[FEATURE_ORDER]

    # ---------------------------------------------------------
    # 5. Validate shape
    # ---------------------------------------------------------
    if data.shape != (1, len(FEATURE_ORDER)):
        raise ValueError(
            f"Expected input shape (1, {len(FEATURE_ORDER)}), "
            f"but received {data.shape}."
        )

    # ---------------------------------------------------------
    # 6. Predict
    # ---------------------------------------------------------
    prediction = model.predict(data)[0]

    return float(prediction)