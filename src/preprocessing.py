import numpy as np
import pandas as pd


FEATURE_ORDER = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "sale_year",
    "house_age",
    "living_to_lot_ratio",
    "living_sqft_per_bedroom",
    "years_since_renovation",
    "total_sqft",
]


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing and feature engineering
    used during model training.
    """

    data = data.copy()

    # ---------------------------------------------------------
    # Date
    # ---------------------------------------------------------

    data["date"] = pd.to_datetime(data["date"])

    data["sale_year"] = data["date"].dt.year

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------

    # 1. House age at time of sale
    data["house_age"] = (
        data["sale_year"] - data["yr_built"]
    )

    # 2. Living area relative to lot size
    data["living_to_lot_ratio"] = (
        data["sqft_living"]
        / data["sqft_lot"].replace(0, np.nan)
    )

    # 3. Living space per bedroom
    data["living_sqft_per_bedroom"] = (
        data["sqft_living"]
        / data["bedrooms"].replace(0, np.nan)
    )

    # 4. Years since renovation
    #
    # Houses with yr_renovated == 0 were never renovated,
    # so we represent this as 0.
    data["years_since_renovation"] = (
        data["sale_year"]
        - data["yr_renovated"].replace(0, np.nan)
    )

    data["years_since_renovation"] = (
        data["years_since_renovation"]
        .fillna(0)
    )

    # 5. Total square footage
    data["total_sqft"] = (
        data["sqft_living"]
        + data["sqft_lot"]
    )

    # ---------------------------------------------------------
    # Handle invalid values
    # ---------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ---------------------------------------------------------
    # Fill engineered missing values
    # ---------------------------------------------------------

    data["living_to_lot_ratio"] = (
        data["living_to_lot_ratio"]
        .fillna(
            data["living_to_lot_ratio"].median()
        )
    )

    data["living_sqft_per_bedroom"] = (
        data["living_sqft_per_bedroom"]
        .fillna(
            data["living_sqft_per_bedroom"].median()
        )
    )

    # ---------------------------------------------------------
    # Remove unused columns
    # ---------------------------------------------------------

    data = data.drop(
        columns=["id", "date"],
        errors="ignore"
    )

    # ---------------------------------------------------------
    # Enforce exact feature order
    # ---------------------------------------------------------

    data = data[FEATURE_ORDER]

    # ---------------------------------------------------------
    # Match training data types
    # ---------------------------------------------------------

    integer_columns = [
        "bedrooms",
        "sqft_living",
        "sqft_lot",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "zipcode",
        "sqft_living15",
        "sqft_lot15",
        "sale_year",
        "house_age",
        "years_since_renovation",
        "total_sqft",
    ]

    for column in integer_columns:
        data[column] = data[column].astype(int)

    return data