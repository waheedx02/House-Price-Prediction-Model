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
        /* ================================
        SIDEBAR
        ================================ */

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }

        /* Sidebar brand */

        .sidebar-brand {
            padding: 0.2rem 0.2rem 1.5rem 0.2rem;
        }

        .sidebar-brand-title {
            font-size: 1.25rem;
            font-weight: 750;
            color: #ffffff;
            letter-spacing: -0.02em;
        }

        .sidebar-brand-subtitle {
            margin-top: 0.2rem;
            color: rgba(255, 255, 255, 0.48);
            font-size: 0.78rem;
        }

        /* Model status */

        .model-status {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.7rem 0.85rem;
            margin-bottom: 1.25rem;
            border-radius: 10px;
            background: rgba(25, 135, 84, 0.10);
            border: 1px solid rgba(25, 135, 84, 0.22);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #39d98a;
            box-shadow: 0 0 10px rgba(57, 217, 138, 0.65);
        }

        .status-text {
            color: #9ee7bf;
            font-size: 0.78rem;
            font-weight: 650;
            letter-spacing: 0.04em;
        }

        /* Section headings */

        .sidebar-section-title {
            margin: 1.25rem 0 0.7rem 0;
            color: rgba(255, 255, 255, 0.52);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }

        /* Model information card */

        .model-info-card {
            padding: 0.9rem;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.07);
        }

        .model-name {
            color: #ffffff;
            font-size: 0.9rem;
            font-weight: 650;
            margin-bottom: 0.8rem;
        }

        .model-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem;
        }

        .model-metric {
            padding: 0.65rem;
            border-radius: 9px;
            background: rgba(255, 255, 255, 0.035);
        }

        .metric-label {
            color: rgba(255, 255, 255, 0.42);
            font-size: 0.68rem;
        }

        .metric-value {
            margin-top: 0.15rem;
            color: rgba(255, 255, 255, 0.9);
            font-size: 0.84rem;
            font-weight: 700;
        }

        /* About section */

        .sidebar-description {
            color: rgba(255, 255, 255, 0.58);
            font-size: 0.78rem;
            line-height: 1.6;
        }

        /* Workflow */

        .workflow-step {
            display: flex;
            align-items: flex-start;
            gap: 0.7rem;
            margin-bottom: 0.8rem;
        }

        .workflow-number {
            flex-shrink: 0;
            width: 25px;
            height: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 7px;
            background: rgba(25, 135, 84, 0.12);
            border: 1px solid rgba(25, 135, 84, 0.20);
            color: #75d6a5;
            font-size: 0.68rem;
            font-weight: 700;
        }

        .workflow-content {
            padding-top: 0.1rem;
        }

        .workflow-title {
            color: rgba(255, 255, 255, 0.82);
            font-size: 0.76rem;
            font-weight: 650;
        }

        .workflow-description {
            margin-top: 0.1rem;
            color: rgba(255, 255, 255, 0.42);
            font-size: 0.7rem;
            line-height: 1.4;
        }
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

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background:
                radial-gradient(
                    circle at 50% -20%,
                    rgba(25, 135, 84, 0.22),
                    transparent 55%
                ),
                linear-gradient(
                    145deg,
                    rgba(25, 135, 84, 0.13),
                    rgba(20, 25, 32, 0.95)
                );
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
        }

        .prediction-card {
            position: relative;
            overflow: hidden;
            margin: 1.2rem 0 0.8rem 0;
            padding: 2rem 2rem 1.7rem 2rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.10);
            background:
                radial-gradient(
                    circle at 50% -20%,
                    rgba(25, 135, 84, 0.22),
                    transparent 55%
                ),
                linear-gradient(
                    145deg,
                    rgba(25, 135, 84, 0.13),
                    rgba(20, 25, 32, 0.95)
                );
            box-shadow:
                0 12px 35px rgba(0, 0, 0, 0.25),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .prediction-card:hover {
            transform: translateY(-2px);
            box-shadow:
                0 16px 42px rgba(0, 0, 0, 0.32),
                0 0 30px rgba(25, 135, 84, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }

        .prediction-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.8rem;
            margin-bottom: 0.9rem;
            border-radius: 999px;
            background: rgba(25, 135, 84, 0.14);
            border: 1px solid rgba(25, 135, 84, 0.28);
            color: #75d6a5;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .prediction-label {
            color: rgba(255, 255, 255, 0.68);
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 0.35rem;
        }

        .prediction-price {
            color: #ffffff;
            font-size: 3.4rem;
            line-height: 1.1;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 1.2rem;
            text-shadow: 0 2px 20px rgba(25, 135, 84, 0.18);
        }

        .prediction-range-label {
            color: rgba(255, 255, 255, 0.52);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 0.65rem;
        }

        .prediction-range-values {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.75rem;
            color: rgba(255, 255, 255, 0.88);
            font-size: 0.95rem;
        }

        .prediction-range-values .range-value {
            font-weight: 700;
        }

        .prediction-range-values .range-separator {
            color: rgba(255, 255, 255, 0.3);
        }

        .prediction-range-bar {
            position: relative;
            height: 7px;
            margin: 1rem auto 0.45rem auto;
            max-width: 500px;
            border-radius: 999px;
            background: linear-gradient(
                90deg,
                rgba(255, 255, 255, 0.10),
                rgba(25, 135, 84, 0.55),
                rgba(255, 255, 255, 0.10)
            );
        }

        .prediction-range-marker {
            position: absolute;
            top: 50%;
            left: 50%;
            width: 15px;
            height: 15px;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            background: #ffffff;
            border: 3px solid #198754;
            box-shadow:
                0 0 0 4px rgba(25, 135, 84, 0.18),
                0 2px 8px rgba(0, 0, 0, 0.3);
        }

        .prediction-range {
            margin-top: 0.2rem;
            color: rgba(255, 255, 255, 0.58);
            font-size: 0.78rem;
        }

        .prediction-range strong {
            color: rgba(255, 255, 255, 0.82);
        }

        .prediction-meta {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.07);
        }

        .prediction-meta-item {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .prediction-meta-label {
            color: rgba(255, 255, 255, 0.42);
            font-size: 0.72rem;
        }

        .prediction-meta-value {
            color: rgba(255, 255, 255, 0.85);
            font-size: 0.85rem;
            font-weight: 600;
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

    # --------------------------------
    # Brand
    # --------------------------------

    st.html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                🏠 House Price Predictor
            </div>
            <div class="sidebar-brand-subtitle">
                Machine learning price estimation
            </div>
        </div>
        """
    )

    # --------------------------------
    # Model status
    # --------------------------------

    st.html(
        """
        <div class="model-status">
            <div class="status-dot"></div>
            <div class="status-text">MODEL READY</div>
        </div>
        """
    )

    # --------------------------------
    # Model
    # --------------------------------

    st.html(
        """
        <div class="sidebar-section-title">
            Model
        </div>

        <div class="model-info-card">

            <div class="model-name">
                CatBoost Regressor
            </div>

            <div class="model-metrics">

                <div class="model-metric">
                    <div class="metric-label">R²</div>
                    <div class="metric-value">0.9121</div>
                </div>

                <div class="model-metric">
                    <div class="metric-label">MAE</div>
                    <div class="metric-value">$63,144</div>
                </div>

                <div class="model-metric">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">$114,844</div>
                </div>

                <div class="model-metric">
                    <div class="metric-label">Features</div>
                    <div class="metric-value">24</div>
                </div>

            </div>

        </div>
        """
    )

    # --------------------------------
    # About
    # --------------------------------

    st.html(
        """
        <div class="sidebar-section-title">
            About this model
        </div>

        <div class="sidebar-description">
            This model estimates King County home sale prices
            using property characteristics, quality indicators,
            and geographic information.
        </div>
        """
    )

    # --------------------------------
    # Workflow
    # --------------------------------

    st.html(
        """
        <div class="sidebar-section-title">
            How it works
        </div>

        <div class="workflow-step">
            <div class="workflow-number">01</div>
            <div class="workflow-content">
                <div class="workflow-title">Enter property details</div>
                <div class="workflow-description">
                    Provide the home's characteristics and location.
                </div>
            </div>
        </div>

        <div class="workflow-step">
            <div class="workflow-number">02</div>
            <div class="workflow-content">
                <div class="workflow-title">Engineer features</div>
                <div class="workflow-description">
                    Derived features are calculated automatically.
                </div>
            </div>
        </div>

        <div class="workflow-step">
            <div class="workflow-number">03</div>
            <div class="workflow-content">
                <div class="workflow-title">Generate prediction</div>
                <div class="workflow-description">
                    CatBoost estimates the property's sale price.
                </div>
            </div>
        </div>
        """
    )

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

                <div class="prediction-badge">
                    ✓ Model Prediction
                </div>

                <div class="prediction-label">
                    Estimated Sale Price
                </div>

                <div class="prediction-price">
                    ${prediction:,.0f}
                </div>

                <div class="prediction-range-label">
                    Typical prediction range
                </div>

                <div class="prediction-range-values">
                    <span class="range-value">
                        ${lower_bound:,.0f}
                    </span>

                    <span class="range-separator">—</span>

                    <span class="range-value">
                        ${upper_bound:,.0f}
                    </span>
                </div>

                <div class="prediction-range-bar">
                    <div class="prediction-range-marker"></div>
                </div>

                <div class="prediction-range">
                    Based on the model's validation error
                </div>

                <div class="prediction-meta">

                    <div class="prediction-meta-item">
                        <span class="prediction-meta-label">
                            Model
                        </span>
                        <span class="prediction-meta-value">
                            CatBoost
                        </span>
                    </div>

                    <div class="prediction-meta-item">
                        <span class="prediction-meta-label">
                            Validation MAE
                        </span>
                        <span class="prediction-meta-value">
                            $63,144
                        </span>
                    </div>

                    <div class="prediction-meta-item">
                        <span class="prediction-meta-label">
                            R²
                        </span>
                        <span class="prediction-meta-value">
                            0.9121
                        </span>
                    </div>

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