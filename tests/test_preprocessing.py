import pandas as pd

from src.preprocessing import create_features, FEATURE_ORDER


def make_sample():
    return pd.DataFrame([{
        "id": 0,
        "date": "20141013",

        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1800,
        "sqft_lot": 5000,
        "floors": 1.5,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,

        "sqft_above": 1500,
        "sqft_basement": 300,

        "yr_built": 1990,
        "yr_renovated": 0,

        "zipcode": 98103,
        "lat": 47.5480,
        "long": -122.2,

        "sqft_living15": 1800,
        "sqft_lot15": 5000,
    }])


def test_feature_count():
    data = create_features(make_sample())

    assert data.shape == (1, 24)


def test_feature_order():
    data = create_features(make_sample())

    assert list(data.columns) == FEATURE_ORDER


def test_no_missing_values():
    data = create_features(make_sample())

    assert data.isnull().sum().sum() == 0


def test_house_age():
    data = create_features(make_sample())

    assert data.loc[0, "house_age"] == 24


def test_years_since_renovation():
    data = create_features(make_sample())

    assert data.loc[0, "years_since_renovation"] == 0


def test_total_sqft():
    data = create_features(make_sample())

    assert data.loc[0, "total_sqft"] == 6800


def test_living_to_lot_ratio():
    data = create_features(make_sample())

    assert data.loc[0, "living_to_lot_ratio"] == 1800 / 5000


def test_living_sqft_per_bedroom():
    data = create_features(make_sample())

    assert data.loc[0, "living_sqft_per_bedroom"] == 600