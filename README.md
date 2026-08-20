# 🏠 House Price Prediction

A machine learning project that predicts house prices using the Kaggle King County house sales dataset.

The project includes a complete workflow from data exploration and feature engineering to model training, evaluation, and a user-friendly Streamlit web application.

---

## 📌 Overview

The goal of this project is to build a reliable house price prediction model while keeping the final user experience simple.

The trained model uses **26 features**, but users do not need to manually enter the engineered features. The application collects the real-world property information and calculates the derived features automatically through the same preprocessing pipeline used during training.

### Prediction pipeline

```text
User Input
    ↓
src/preprocessing.py
    ↓
Feature Engineering
    ↓
CatBoostRegressor
    ↓
src/predict.py
    ↓
Predicted House Price
```

---

## 📊 Model Performance

The final CatBoost model was evaluated using an 80/20 train-test split.

| Metric | Result |
|---|---:|
| MAE | **$63,020.69** |
| RMSE | **$113,638.49** |
| R² | **0.9140** |

Dataset split:

- **Original rows:** 21,613
- **Cleaned rows:** 21,612
- **Training samples:** 17,289
- **Testing samples:** 4,323

The model achieved an R² of **0.9140**, meaning it explains approximately 91.4% of the variance in the test-set house prices.

---

## 🤖 Model Development

Several models were evaluated during development.

### Linear Regression — Baseline

| Metric | Result |
|---|---:|
| MAE | $125,890.46 |
| RMSE | $210,245.75 |
| R² | 0.7055 |

### Tree-Based Model

| Metric | Result |
|---|---:|
| MAE | $72,707.68 |
| RMSE | $145,823.87 |
| R² | 0.8583 |

### Final CatBoost Model

| Metric | Result |
|---|---:|
| MAE | **$63,020.69** |
| RMSE | **$113,638.49** |
| R² | **0.9140** |

Feature engineering and CatBoost significantly improved the model compared with the initial linear regression baseline.

---

## 🧠 Feature Engineering

The project creates additional features from the original dataset.

The feature engineering logic is centralized in:

```text
src/preprocessing.py
```

### Engineered features

| Feature | Formula |
|---|---|
| `sale_year` | Year extracted from `date` |
| `sale_month` | Month extracted from `date` |
| `house_age` | `sale_year - yr_built` |
| `years_since_renovation` | `sale_year - yr_renovated` |
| `total_sqft` | `sqft_above + sqft_basement` |
| `living_to_lot_ratio` | `sqft_living / sqft_lot` |
| `living_sqft_per_bedroom` | `sqft_living / bedrooms` |
| `bathrooms_per_bedroom` | `bathrooms / bedrooms` |

Special handling is applied to invalid or infinite values, and missing values in the two per-bedroom engineered features are filled using their median.

The raw `id` and `date` columns are removed before the data is passed to the model.

---

## 🔎 Model Feature Importance

Feature importance was examined using the final CatBoost model.

The most important features included:

| Feature | Importance |
|---|---:|
| `lat` | 22.749 |
| `long` | 13.452 |
| `grade` | 10.872 |
| `zipcode` | 9.534 |
| `total_sqft` | 8.268 |
| `sqft_living` | 8.110 |
| `sqft_above` | 4.014 |
| `sqft_living15` | 3.751 |
| `view` | 3.600 |
| `waterfront` | 3.488 |

This shows that **location, construction grade, and property size** are particularly influential in the model.

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── raw/
│       └── kc_house_data.csv
│
├── models/
│   └── house_price_model.cbm
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── src/
│   ├── predict.py
│   └── preprocessing.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Main components

**`notebooks/01_exploration.ipynb`**

Contains the data exploration, cleaning, feature engineering, model experiments, evaluation, and feature importance analysis.

**`src/preprocessing.py`**

Contains the feature engineering and preprocessing logic used during prediction.

**`src/predict.py`**

Loads the trained CatBoost model and provides the `predict_price()` function used by the application.

**`models/house_price_model.cbm`**

The trained CatBoost model.

**`app/app.py`**

The Streamlit user interface.

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Matplotlib
- Seaborn
- Jupyter
- Streamlit

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd House-Price-Prediction
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Streamlit Application

From the project root:

```bash
streamlit run app/app.py
```

The application will open in your browser.

The application uses the trained model located at:

```text
models/house_price_model.cbm
```

---

## 🖥️ Streamlit Interface

The application is organized into four sections to keep the prediction form easy to use.

### 🛏️ Basics

Users provide:

- Bedrooms
- Bathrooms
- Floors
- Waterfront status
- View quality

### 📐 Size

Users provide:

- Living area
- Lot size
- Above-ground area
- Basement area
- Average living area of nearby homes
- Average lot size of nearby homes

### ⭐ Quality

Users provide:

- Overall condition
- Construction/design grade

### 📍 Location & Sale Date

Users provide:

- Zip code
- Latitude
- Longitude
- Year built
- Renovation information
- Sale date

The application automatically calculates the engineered features instead of asking users to manually provide them.

---

## 🔄 How Prediction Works

When the user clicks **Predict House Price**:

1. Streamlit collects the property information.
2. The application creates the raw feature dictionary.
3. `src/predict.py` passes the data to `src/preprocessing.py`.
4. The preprocessing pipeline creates the engineered features.
5. The processed data is passed to the trained CatBoost model.
6. CatBoost returns the predicted price.
7. The application displays the estimated price.

Keeping preprocessing in a dedicated module helps ensure that the prediction pipeline remains consistent with the training pipeline.

---

## 🧪 Data Cleaning

The original dataset contains 21,613 records.

During exploration, an extreme outlier containing **33 bedrooms** was identified and removed.

```text
Original rows: 21,613
Cleaned rows: 21,612
```

The remaining data was used for model development and evaluation.

---

## ⚠️ Limitations

This project has several limitations:

- The model is trained on historical King County housing data.
- Predictions may not generalize to other geographic markets.
- Location-related features such as latitude, longitude, and zipcode have a strong influence on predictions.
- The quality of user-provided location and property information directly affects the prediction.
- The model should not be considered a professional property appraisal or financial valuation.

---

## 🔮 Future Improvements

Potential improvements include:

- More robust cross-validation
- Automated hyperparameter optimization
- Better geospatial feature engineering
- Prediction intervals and uncertainty estimates
- Automated testing for the preprocessing and prediction pipeline
- Model monitoring
- CI/CD
- Cloud deployment
- Improved input validation
- A more detailed model explainability section

---

## 📚 Dataset

This project uses the **King County House Sales** dataset commonly distributed through Kaggle.

The dataset contains historical house sale information including property size, location, condition, construction grade, bedrooms, bathrooms, and other characteristics.

---

## 👨‍💻 Project Goal

This project was built as a practical machine learning portfolio project with an emphasis on:

- Data exploration
- Data cleaning
- Feature engineering
- Model comparison
- Gradient boosting
- Model evaluation
- Reusable preprocessing
- Production-style prediction code
- A user-friendly machine learning application

