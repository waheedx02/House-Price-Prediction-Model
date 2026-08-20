import numpy as np
import pandas as pd

def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Convert date
    data["date"] = pd.to_datetime(data["date"])

    # Date features
    data["sale_year"] = data["date"].dt.year
    data["sale_month"] = data["date"].dt.month

    # House age
    data["house_age"] = data["sale_year"] - data["yr_built"]

    # Years since renovation
    data["years_since_renovation"] = np.where(
        (data["yr_renovated"] == 0)
        | (data["yr_renovated"] > data["sale_year"]),
        0,
        data["sale_year"] - data["yr_renovated"],
    )

    # Total square footage
    data["total_sqft"] = (
        data["sqft_above"] + data["sqft_basement"]
    )

    # Ratios
    data["living_to_lot_ratio"] = (
        data["sqft_living"]
        / data["sqft_lot"].replace(0, np.nan)
    )

    data["living_sqft_per_bedroom"] = (
        data["sqft_living"]
        / data["bedrooms"].replace(0, np.nan)
    )

    data["bathrooms_per_bedroom"] = (
        data["bathrooms"]
        / data["bedrooms"].replace(0, np.nan)
    )

    # Handle invalid values
    data = data.replace([np.inf, -np.inf], np.nan)

    # Fill missing engineered values
    data["living_sqft_per_bedroom"] = (
        data["living_sqft_per_bedroom"]
        .fillna(data["living_sqft_per_bedroom"].median())
    )

    data["bathrooms_per_bedroom"] = (
        data["bathrooms_per_bedroom"]
        .fillna(data["bathrooms_per_bedroom"].median())
    )

    # Remove unused columns
    data = data.drop(columns=["id", "date"])

    # Match the feature dtypes used during model training
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
        "sale_month",
        "house_age",
        "years_since_renovation",
        "total_sqft",
    ]

    for column in integer_columns:
        data[column] = data[column].astype(int)

    return data