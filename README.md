# Vehicle-Price-Prediction-using-Machine-Learning
A machine learning project that predicts vehicle prices based on features such as make, model, mileage, engine size, and age using Ridge, Random Forest, and Gradient Boosting models.
# 🚗 Vehicle Price Prediction using Machine Learning

This project focuses on building a predictive model to estimate the **price of vehicles** based on their specifications, including make, model, year, engine details, fuel type, and more. It utilizes machine learning regression algorithms and data preprocessing techniques to create a robust price prediction system.

---

## 🎯 Objective
To develop a machine learning model capable of predicting **vehicle prices** using available data on vehicle features and specifications. The project explores data cleaning, feature extraction, and model optimization for accurate price estimation.

---

## 📘 Dataset Overview

### 📍 Source
A dataset containing information on various vehicles, including their specifications, configurations, and prices.

### 🧩 Description of Columns
| Feature | Description |
|----------|-------------|
| name | Full name of the vehicle (make, model, trim) |
| description | Brief description with key selling points |
| make | Manufacturer (e.g., Ford, Toyota, BMW) |
| model | Model name of the vehicle |
| year | Year of manufacture |
| price | Vehicle price in USD (Target variable) |
| engine | Engine details (type, displacement, etc.) |
| cylinders | Number of engine cylinders |
| fuel | Type of fuel (Gasoline, Diesel, Electric, etc.) |
| mileage | Vehicle mileage in miles |
| transmission | Transmission type (Automatic, Manual) |
| trim | Trim level indicating feature package |
| body | Body type (SUV, Sedan, Pickup, etc.) |
| doors | Number of doors |
| exterior_color | Vehicle’s exterior color |
| interior_color | Vehicle’s interior color |
| drivetrain | Drivetrain type (AWD, FWD, RWD, etc.) |

---

## ⚙️ Project Workflow

### 1️⃣ Data Cleaning & Preprocessing
- Removed irrelevant columns (`description`, `name`, `exterior_color`, `interior_color`)
- Extracted **engine displacement** from engine details (e.g., “3.5L” → 3.5)
- Replaced missing numerical values with **median** and categorical values with **mode**
- Derived **vehicle age** from the manufacturing year
- One-hot encoded categorical variables using `pd.get_dummies`
- Scaled numerical features with **StandardScaler**

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized price distribution using histograms  
- Examined feature correlations using a heatmap  
- Created scatter and box plots:
  - Age vs. Price  
  - Price by Vehicle Make (Top 6 Manufacturers)  

### 3️⃣ Model Training and Hyperparameter Tuning
Trained and evaluated multiple regression models for performance comparison:

| Model | Description | Evaluation Metric |
|--------|-------------|-------------------|
| Ridge Regression | Linear model with L2 regularization | MAE, RMSE, R² |
| Random Forest Regressor | Ensemble tree-based model | MAE, RMSE, R² |
| Gradient Boosting Regressor | Boosting model with GridSearchCV tuning | MAE, RMSE, R² |

Hyperparameter tuning performed with **GridSearchCV** to identify optimal model parameters.

---

## 📊 Model Evaluation

| Model | MAE | RMSE | R² |
|--------|-----|------|----|
| Ridge Regression | ~2800 | ~3700 | 0.78 |
| Random Forest (Tuned) | ~2200 | ~3100 | **0.88** |
| Gradient Boosting (Tuned) | ~2100 | ~3000 | **0.90** |

*(Values are approximate and may vary based on data split and tuning.)*

Visualized **Actual vs Predicted Prices** for Gradient Boosting using a scatter plot.

---

## 🧠 Technologies Used
- **Python 3.8+**
- **Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `pickle`

---
