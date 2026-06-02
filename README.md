# Bike Sharing Demand Prediction

## 📌 Project Overview

This project aims to predict the total number of bike rentals (`cnt`) using weather and temporal features from the **Bike Sharing Dataset**. It is a **supervised regression problem** where we analyze feature relationships, preprocess data, and evaluate multiple machine learning models.

## 📊 Dataset

- **Source**: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) (day.csv)
- **Size**: 731 daily records (2011–2012)
- **Target variable**: `cnt` (total bike rentals)
- **Features**:
  - Temporal: `season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday`
  - Weather: `weathersit`, `temp`, `atemp`, `hum`, `windspeed`

> Note: `casual` and `registered` were removed as they are leakage features.

## 🔍 Exploratory Data Analysis (EDA)

- **Temperature** shows strong positive correlation with bike demand.
- **Humidity** and **windspeed** have negative relationships.
- Seasonal patterns: higher rentals in summer/fall.
- No missing values or duplicates found.

![Feature Correlation Heatmap](images/correlation.png)

## ⚙️ Data Preprocessing

- Dropped `instant` and `dteday` (irrelevant)
- Removed `casual` and `registered` (leakage)
- Scaled numerical features (`temp`, `atemp`, `hum`, `windspeed`) using `StandardScaler`
- Train/test split: 80/20 (random and temporal splits compared)

## 🤖 Models Evaluated

| Model                 | RMSE   | MAE    | R²     |
|-----------------------|--------|--------|--------|
| Linear Regression     | 831.29 | 617.39 | 0.8277 |
| Linear SVR            | 891.09 | 656.34 | 0.8020 |
| RBF SVR               | 1766.68| 1495.73| 0.2216 |
| Decision Tree         | 887.57 | 596.91 | 0.8035 |
| **Random Forest**     | **677.09** | **–** | **0.8857** |

Random Forest performed best after hyperparameter tuning (n_estimators=100).

## 📈 Feature Importance

- **`temp`** and **`yr`** are the most important predictors.
- Humidity, windspeed, and weekday have low impact.
- Confirmed by both **Random Forest built-in importance** and **permutation importance**.

## 📂 Project Structure
