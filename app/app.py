import sys

from pathlib import Path

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Application imports
# ---------------------------------------------------------

import streamlit as st
import pandas as pd

from datetime import date
from src.predict import predict_price

# ---------------------------------------------------------
# Global Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .hero {
            padding: 1.8rem 2rem;
            border-radius: 18px;
            margin-bottom: 1.5rem;
            background: linear-gradient(
                135deg,
                rgba(15, 81, 50, 0.12),
                rgba(20, 108, 67, 0.04)
            );
            border: 1px solid rgba(20, 108, 67, 0.18);
            display: flex;
            align-items: center;
            gap: 1.2rem;
        }

        .hero-icon {
            font-size: 3.2rem;
            line-height: 1;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.4rem;
            font-weight: 750;
        }

        .hero p {
            margin: 0.35rem 0 0 0;
            font-size: 1.05rem;
            opacity: 0.75;
        }
        .section-spacing {
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        .section-divider {
            margin-top: 2rem;
            margin-bottom: 2rem;
            border: none;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
        }
        .predict-button {
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            font-size: 1.05rem;
            font-weight: 600;
            width: 100%;
            cursor: pointer;
        }

        .prediction-card {
            padding: 2rem;
            border-radius: 18px;
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            text-align: center;
            margin-top: 1.5rem;
        }

        .prediction-label {
            font-size: 1rem;
            opacity: 0.7;
            margin-bottom: 0.4rem;
        }

        .prediction-price {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .prediction-range {
            font-size: 0.95rem;
            opacity: 0.7;
        }

        .result-section {
            margin-top: 2rem;
        }
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 10px;
            min-height: 3rem;
            font-size: 1.05rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Global Constants
# ---------------------------------------------------------

MODEL_MAE = 63144.44

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.title("🏠 House Price Predictor")

    st.caption(
        "Machine learning application for predicting "
        "King County house prices."
    )

    st.divider()

    # Model information
    st.subheader("🤖 Model")

    st.markdown(
        """
        **CatBoost Regressor**

        A gradient boosting model designed to capture
        nonlinear relationships between property
        characteristics and sale prices.
        """
    )

    st.divider()

    # Model performance
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("R²", "0.9121")
        st.metric("MAE", "$63.14K")

    with col2:
        st.metric("RMSE", "$114.8K")
        st.metric("CV RMSE", "$117.1K")

    st.caption(
        "Evaluation performed on the held-out test set."
    )

    st.divider()

    # Technical information
    st.subheader("🔧 Tech Stack")

    st.markdown(
        """
        **Python** · **Pandas** · **NumPy**  
        **Scikit-learn** · **CatBoost** · **Streamlit**
        """
    )

    st.divider()

    st.caption("House Price Prediction • Machine Learning Project")

# ---------------------------------------------------------
# Hero / Header
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🏠</div>
        <div>
            <h1>House Price Predictor</h1>
            <p>
                Estimate the market value of a King County home
                using a trained machine learning model.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Property Details
# ---------------------------------------------------------

st.markdown("## 🏡 Property Details")

st.caption(
    "Enter the information you know about the property. "
    "Derived features are calculated automatically."
)

with st.form("house_form"):

    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Basic Information
    # -----------------------------------------------------

    st.markdown("### 🛏️ Basic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=15,
            value=3,
            step=1,
        )

    with col2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.25,
            help="Examples: 1.5, 2.25, 3.5",
        )

    with col3:
        floors = st.selectbox(
            "Floors",
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            index=1,
        )

    col1, col2 = st.columns(2)

    with col1:
        waterfront = st.radio(
            "Waterfront Property",
            ["No", "Yes"],
            horizontal=True,
        )

    with col2:
        view = st.slider(
            "View Quality",
            min_value=0,
            max_value=4,
            value=0,
            help="0 = No view · 4 = Excellent view",
        )

    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Property Size
    # -----------------------------------------------------

    st.markdown("### 📐 Property Size")

    col1, col2 = st.columns(2)

    with col1:
        sqft_living = st.number_input(
            "Living Area (sqft)",
            min_value=200,
            max_value=20000,
            value=1800,
            step=50,
            help="Total interior living area.",
        )

    with col2:
        sqft_lot = st.number_input(
            "Lot Size (sqft)",
            min_value=200,
            max_value=500000,
            value=5000,
            step=100,
        )

    col1, col2 = st.columns(2)

    with col1:
        sqft_above = st.number_input(
            "Above-Ground Area (sqft)",
            min_value=0,
            max_value=20000,
            value=1500,
            step=50,
        )

    with col2:
        sqft_basement = st.number_input(
            "Basement Area (sqft)",
            min_value=0,
            max_value=10000,
            value=300,
            step=50,
        )

    col1, col2 = st.columns(2)

    with col1:
        sqft_living15 = st.number_input(
            "Avg. Neighbor Living Area (sqft)",
            min_value=200,
            max_value=20000,
            value=1800,
            step=50,
            help="Average living area of the 15 nearest houses.",
        )

    with col2:
        sqft_lot15 = st.number_input(
            "Avg. Neighbor Lot Size (sqft)",
            min_value=200,
            max_value=500000,
            value=5000,
            step=100,
            help="Average lot size of the 15 nearest houses.",
        )

    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Quality & Condition
    # -----------------------------------------------------

    st.markdown("### ⭐ Quality & Condition")

    col1, col2 = st.columns(2)

    with col1:
        condition = st.slider(
            "Overall Condition",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Poor · 5 = Excellent",
        )

    with col2:
        grade = st.slider(
            "Construction & Design Grade",
            min_value=1,
            max_value=13,
            value=7,
            help="1–3 = Low quality · 7 = Average · 11–13 = High-end",
        )

    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Location & Sale Information
    # -----------------------------------------------------

    st.markdown("### 📍 Location & Sale Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        zipcode = st.number_input(
            "Zip Code",
            min_value=98001,
            max_value=98199,
            value=98103,
            step=1,
        )

    with col2:
        lat = st.number_input(
            "Latitude",
            min_value=47.0,
            max_value=48.0,
            value=47.5480,
            step=0.0001,
            format="%.4f",
        )

    with col3:
        long_ = st.number_input(
            "Longitude",
            min_value=-123.0,
            max_value=-121.0,
            value=-122.2,
            step=0.0001,
            format="%.4f",
        )

    col1, col2 = st.columns(2)

    with col1:
        yr_built = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=date.today().year,
            value=1990,
            step=1,
        )

    with col2:
        yr_renovated = st.number_input(
            "Year Renovated",
            min_value=0,
            max_value=date.today().year,
            value=0,
            step=1,
            help="Enter 0 if the property has never been renovated.",
        )

    sale_date = st.date_input(
        "Sale Date",
        value=date.today(),
        help="The model uses the year of sale as a feature.",
    )

    st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
    st.markdown("---")

    submitted = st.form_submit_button(
        "🔮 Predict House Price",
        use_container_width=True,
    )

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    raw_input = {
        "id": 0,
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
        # -------------------------------------------------
        # Generate Prediction
        # -------------------------------------------------

        with st.spinner("🏠 Analyzing property and generating prediction..."):
            prediction = predict_price(raw_input)

        # -------------------------------------------------
        # Calculate Typical Prediction Range
        # -------------------------------------------------

        lower_bound = max(0, prediction - MODEL_MAE)
        upper_bound = prediction + MODEL_MAE

        # -------------------------------------------------
        # Prediction Result
        # -------------------------------------------------

        st.markdown(
            "<div class='result-section'></div>",
            unsafe_allow_html=True,
        )

        st.markdown("## 💰 Prediction Result")

        st.html(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Estimated Sale Price</div>
                <div class="prediction-price">${prediction:,.0f}</div>
                <div class="prediction-range">
                    Typical model range:
                    <strong>${lower_bound:,.0f} – ${upper_bound:,.0f}</strong>
                </div>
            </div>
            """
        )

        st.caption(
            "The range is based on the model's validation MAE and represents "
            "a typical prediction error, not a guaranteed price range."
        )

        # -------------------------------------------------
        # Property Summary
        # -------------------------------------------------

        with st.expander("📋 View Property Summary"):

            summary_col1, summary_col2, summary_col3 = st.columns(3)

            with summary_col1:
                st.metric("Bedrooms", bedrooms)
                st.metric("Bathrooms", bathrooms)
                st.metric("Living Area", f"{sqft_living:,} sqft")

            with summary_col2:
                st.metric("Lot Size", f"{sqft_lot:,} sqft")
                st.metric("Year Built", yr_built)
                st.metric("Grade", grade)

            with summary_col3:
                st.metric("Zip Code", zipcode)
                st.metric("Condition", condition)
                st.metric(
                    "Waterfront",
                    "Yes" if waterfront == "Yes" else "No",
                )

    except Exception as e:
        st.error(f"Prediction failed: {e}")