import sys
from pathlib import Path
from datetime import date

import streamlit as st

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Application imports
# ---------------------------------------------------------

from src.predict import predict_price


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