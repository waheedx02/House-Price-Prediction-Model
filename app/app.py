"""
House Price Predictor — Streamlit UI
=====================================

Single-page interface for the King County house price prediction
project. Sections are organized as collapsible panels (Basics, Size,
Quality, Location & Sale Date) stacked on one page rather than separate
tabs — click a section header to expand/collapse it.

This file does NOT perform any feature engineering itself. It collects
raw, real-world property details from the user and passes them straight
to the trained pipeline via:

    from src.predict import predict_price

`predict_price()` is responsible for running the input through
`src/preprocessing.py` (feature engineering) and the CatBoost model
loaded from `models/house_price_model.cbm`.

Run from the project root with:

    streamlit run app/app.py

ASSUMPTION TO VERIFY
---------------------
The exact input shape `predict_price()` expects (a single-row DataFrame
vs. a dict, exact column names, and how it wants the "date" field
formatted) was not specified. This file assumes `predict_price()` takes
a single-row pandas DataFrame using the original King County dataset's
raw column names (see `raw_input` below), including a placeholder `id`
column since `preprocessing.py` is described as dropping `id` before
modeling. Adjust the `raw_input` construction if your actual
`predict_price()` signature differs.
"""

import sys
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

# Ensure the project root (parent of `app/`) is importable so
# `from src.predict import predict_price` works regardless of how
# `streamlit run app/app.py` sets up sys.path.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import predict_price  # noqa: E402

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.markdown(
    """
    <style>
    .main-header { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.1rem; }
    .price-box {
        background: linear-gradient(135deg, #0f5132 0%, #146c43 100%);
        color: white; font-size: 2.4rem; font-weight: 700;
        padding: 1.4rem 1.6rem; border-radius: 14px; text-align: center;
        margin: 0.6rem 0 1.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-header'>🏠 House Price Predictor</div>", unsafe_allow_html=True)
st.caption(
    "Enter the property's details below. Engineered fields like house "
    "age and per-bedroom ratios are calculated automatically by the "
    "preprocessing pipeline — you don't need to enter them."
)

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
This app predicts King County home sale prices using a trained
**CatBoostRegressor** model.
"""
    )

with st.form("house_form"):
    with st.expander("🛏️ Basics", expanded=True):
        c1, c2, c3 = st.columns(3)
        bedrooms = c1.number_input("Bedrooms", min_value=0, max_value=15, value=3, step=1)
        bathrooms = c2.number_input(
            "Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.25,
            help="Count half/quarter baths, e.g. 2.25",
        )
        floors = c3.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=1)

        c4, c5 = st.columns(2)
        waterfront = c4.radio("Waterfront property?", ["No", "Yes"], horizontal=True)
        view = c5.slider("View quality", 0, 4, 0, help="0 = No view, 4 = Excellent view")

    with st.expander("📐 Size", expanded=False):
        c1, c2 = st.columns(2)
        sqft_living = c1.number_input("Living area (sqft)", min_value=200, max_value=20000, value=1800, step=50)
        sqft_lot = c2.number_input("Lot size (sqft)", min_value=200, max_value=500000, value=5000, step=100)

        c3, c4 = st.columns(2)
        sqft_above = c3.number_input("Above-ground living area (sqft)", min_value=200, max_value=20000, value=1500, step=50)
        sqft_basement = c4.number_input(
            "Basement area (sqft)", min_value=0, max_value=10000, value=300, step=50,
            help="Enter 0 if there's no basement",
        )

        c5, c6 = st.columns(2)
        sqft_living15 = c5.number_input(
            "Avg. living area of the 15 nearest neighboring houses (sqft)",
            min_value=200, max_value=20000, value=1800, step=50,
        )
        sqft_lot15 = c6.number_input(
            "Avg. lot size of the 15 nearest neighboring houses (sqft)",
            min_value=200, max_value=500000, value=5000, step=100,
        )

    with st.expander("⭐ Quality", expanded=False):
        c1, c2 = st.columns(2)
        condition = c1.slider("Overall condition", 1, 5, 3, help="1 = Poor, 5 = Excellent")
        grade = c2.slider(
            "Construction & design grade", 1, 13, 7,
            help="1–3 = Low quality  •  7 = Average  •  11–13 = High-end custom build",
        )

    with st.expander("📍 Location & Sale Date", expanded=False):
        c1, c2, c3 = st.columns(3)
        zipcode = c1.number_input("Zip code", min_value=98001, max_value=98199, value=98103, step=1)
        lat = c2.number_input("Latitude", min_value=47.0, max_value=48.0, value=47.5480, step=0.0001, format="%.4f")
        long_ = c3.number_input("Longitude", min_value=-123.0, max_value=-121.0, value=-122.2, step=0.0001, format="%.4f")
        st.caption("Not sure of the exact latitude/longitude? Look up the address on Google Maps and right-click → \"What's here\".")

        c4, c5 = st.columns(2)
        yr_built = c4.number_input("Year built", min_value=1900, max_value=date.today().year, value=1990, step=1)
        renovated = c5.checkbox("This house has been renovated")
        yr_renovated = 0
        if renovated:
            yr_renovated = st.number_input(
                "Year renovated", min_value=1900, max_value=date.today().year, value=2010, step=1,
            )

        sale_date = st.date_input("Sale date", value=date.today())

    submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

if submitted:
    if sqft_above + sqft_basement != sqft_living:
        st.info(
            f"Note: above-ground + basement sqft ({sqft_above + sqft_basement:,}) "
            f"doesn't exactly match living area ({sqft_living:,}). That's okay — "
            "the prediction still uses the numbers exactly as you entered them."
        )

    # Raw, unengineered input using the original dataset's column names.
    # `predict_price()` is expected to run this through preprocessing.py
    # itself — see the module docstring for the assumption this makes.
    raw_input = {
        "id": 0,  # placeholder — preprocessing.py drops this column before modeling
        "date": pd.to_datetime(sale_date),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": 1 if waterfront == "Yes" else 0,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "zipcode": zipcode,
        "lat": lat,
        "long": long_,
        "sqft_living15": sqft_living15,
        "sqft_lot15": sqft_lot15,
    }

    try:
        prediction = predict_price(raw_input)
        # predict_price() may return a scalar, or an array/Series of length 1.
        if hasattr(prediction, "__len__") and not isinstance(prediction, (int, float)):
            prediction = prediction[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.markdown("### 💰 Estimated Price")
    st.markdown(f"<div class='price-box'>${prediction:,.0f}</div>", unsafe_allow_html=True)