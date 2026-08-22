# ---------------------------------------------------------
# Application imports
# ---------------------------------------------------------

import streamlit as st
import sys

from pathlib import Path
from datetime import date
from src.predict import predict_price

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    # Feature engineering
    st.subheader("🧠 Feature Engineering")

    st.markdown(
        """
        The model uses engineered features including:

        - **House age**
        - **Living-to-lot ratio**
        - **Living area per bedroom**
        - **Years since renovation**
        - **Total square footage**

        These are calculated automatically from the
        property information entered below.
        """
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
# Quick model summary
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "CatBoost",
    )

with col2:
    st.metric(
        "R² Score",
        "0.9121",
    )

with col3:
    st.metric(
        "Test MAE",
        "$63.1K",
    )

st.markdown(
    """
    <div style="height: 0.8rem;"></div>
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

    st.markdown("---")

    submitted = st.form_submit_button(
        "🔮 Predict House Price",
        use_container_width=True,
    )