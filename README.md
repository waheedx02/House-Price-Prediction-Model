# 🏠 House Price Prediction

An end-to-end machine learning project that predicts house prices using the **King County Housing Dataset**. The project covers the complete ML workflow, from data preprocessing and feature engineering to model training, evaluation, and deployment with a **Streamlit** web application.

---

<h2>Screenshots</h2>

<img src="screenshots/Capture-1.PNG" width="700">

<img src="screenshots/Capture-2.PNG" width="700">

<img src="screenshots/Capture-3.PNG" width="700">

## 📌 Project Overview

This project was built to practice the complete machine learning development process using Python and scikit-learn.

The workflow includes:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model persistence with Joblib
- Interactive prediction using Streamlit

The final model is a **Random Forest Regressor**, which significantly outperformed the initial Linear Regression baseline.

---

## 🚀 Features

- Predict house prices from property information
- Feature engineering for improved model performance
- Random Forest Regression model
- Interactive Streamlit web interface
- Model saving and loading using Joblib
- Clean and organized project structure

---

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── app/
│   ├── app.py          # Streamlit interface
│   ├── train.py        # Model training
│   └── predict.py      # Prediction logic
│
├── data/
│   └── kc_house_data.csv
│
├── model/
│   └── house_price_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Feature Engineering

Additional features were created from the original dataset to improve model performance.

Engineered features include:

- Total Rooms
- Living Quality
- Square Feet per Room
- House Age

The target variable (`price`) was transformed using `log1p()` before training and converted back during prediction.

---

## 📊 Models Compared

| Model | MAE | R² Score |
|-------|------------:|---------:|
| Linear Regression | ~$114,574 | 0.68 |
| Random Forest Regressor | **~$71,374** | **0.88** |

The Random Forest model achieved significantly better performance and was selected as the final model.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/waheed-x02/House-Price-Prediction.git
```

Navigate to the project

```bash
cd House-Price-Prediction
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

If you want to retrain the model:

```bash
python app/train.py
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app/app.py
```

---

## 📈 Example Prediction

The application allows users to enter:

- Property details
- House size
- House quality
- Optional advanced information

The trained model then estimates the selling price of the house.

---

## 📚 What I Learned

Through this project I learned how to:

- Work with real-world datasets
- Perform feature engineering
- Train regression models
- Evaluate models using MAE, MSE, and R²
- Compare multiple machine learning models
- Save and load trained models
- Build an interactive ML application with Streamlit
- Organize an end-to-end machine learning project

---

If you enjoyed this project or have suggestions for improvement, feel free to open an issue or submit a pull request.