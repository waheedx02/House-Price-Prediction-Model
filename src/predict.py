from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.preprocessing import create_features


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "house_price_model.cbm"
)

model = CatBoostRegressor()
model.load_model(MODEL_PATH)


def predict_price(features: dict) -> float:
    data = pd.DataFrame([features])

    data = create_features(data)

    prediction_data = Pool(
        data=data,
        cat_features=["zipcode"]
    )

    prediction = model.predict(prediction_data)[0]

    return float(prediction)