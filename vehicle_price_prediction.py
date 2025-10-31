
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re

# %%
# Load the dataset
df = pd.read_csv("D:/zoology download/Projects-20240722T093004Z-001/Projects/vehicle_price_prediction/Vehicle Price Prediction/dataset.csv")
# Inspect the first few rows and data types
print(df.head(3))
print("\nData types and missing values:")
print(df.dtypes)
print(df.isnull().sum().sort_values(ascending=False))
print("\nDataset shape:", df.shape)
# %%
# Drop exact duplicates to avoid bias
df = df.drop_duplicates()
print("After dropping duplicates, shape:", df.shape)
# Drop entries without a price (target variable) or with invalid prices (<= 0)
df = df[df['price'].notnull() & (df['price'] > 0)]
print("After removing missing/zero prices, shape:", df.shape)
# %%
# Drop text fields that are not needed
df = df.drop(columns=['description', 'name', 'interior_color', 'exterior_color'])
# %%
# Extract numeric displacement from 'engine' string (e.g., "6.2L" -> 6.2)
def extract_displacement(engine_str):
    if isinstance(engine_str, str):
        match = re.search(r'(\d+\.\d+|\d+)L', engine_str)
        if match:
            return float(match.group(1))
    return np.nan
df['engine_displacement'] = df['engine'].apply(extract_displacement)
# Fill missing displacement with median value
df['engine_displacement'].fillna(df['engine_displacement'].median(), inplace=True)
# Fill missing numeric features
df['cylinders'] = df['cylinders'].fillna(df['cylinders'].median())
df['mileage'] = df['mileage'].fillna(df['mileage'].median())
df['doors'] = df['doors'].fillna(df['doors'].mode()[0])
# Fill missing categorical features with mode
df['fuel'] = df['fuel'].fillna(df['fuel'].mode()[0])
df['transmission'] = df['transmission'].fillna(df['transmission'].mode()[0])
df['body'] = df['body'].fillna(df['body'].mode()[0])
df['trim'] = df['trim'].fillna('') # Just fill trim with empty string if missing
# Drop original 'engine' and 'year' columns after extraction
df = df.drop(columns=['engine'])
# Compute vehicle age from year (assuming current year is 2024)
df['age'] = 2024- df['year']
df = df.drop(columns=['year', 'trim']) # Drop 'trim' for simplicity
# Final check for missing values
print("Missing values after cleaning:")
print(df.isnull().sum())
# %%
 # Separate features and target
X_raw = df.drop(columns=['price'])
y = df['price']
# One-hot encode categorical features
X_encoded = pd.get_dummies(X_raw, drop_first=True)
print("Features after encoding:", X_encoded.shape)
print("Sample feature columns:", X_encoded.columns.tolist()[:10])
# %%
#EDA
# Price distribution
plt.figure(figsize=(6,4))
sns.histplot(df['price'], bins=30, kde=True, color='orange')
plt.title("Distribution of Vehicle Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()
# %%
# Correlation heatmap of numeric features
numeric_cols = ['price', 'cylinders', 'mileage', 'doors', 'engine_displacement', 'age']
corr = df[numeric_cols].corr()
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
# %%
# Scatter plot: Age vs. Price
plt.figure(figsize=(6,4))
sns.scatterplot(x='age', y='price', data=df)
plt.title("Vehicle Age vs. Price")
plt.xlabel("Age (years)")
plt.ylabel("Price")
plt.show()
# %%
# Boxplot: Price by Make (top 6 makes)
top_makes = df['make'].value_counts().nlargest(6).index
plt.figure(figsize=(6,4))
sns.boxplot(x='make', y='price', data=df[df['make'].isin(top_makes)])
plt.title("Price by Make (Top 6 Makes)")
plt.xticks(rotation=45)
plt.xlabel("Make")
plt.ylabel("Price")
plt.show()
# %%
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
# %%
# Identify numeric columns for scaling
numeric_features = ['cylinders','mileage','doors','engine_displacement','age']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
# %%
# Scale numeric columns
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
print("Scaled feature sample:")
print(X_train_scaled[numeric_features].head(3))
# %%
#hyperparameter tuning
# Random Forest Grid Search
param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
gs_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='r2')
gs_rf.fit(X_train_scaled, y_train)
print("Best RandomForest params:", gs_rf.best_params_)
# %%
# Gradient Boosting Grid Search
param_grid_gb = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
gs_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=3, scoring='r2')
gs_gb.fit(X_train_scaled, y_train)
print("Best GradientBoosting params:", gs_gb.best_params_)
# %%
# Evaluate tuned models on test set
rf_best = gs_rf.best_estimator_
gb_best = gs_gb.best_estimator_
for model, name in [(rf_best, 'RandomForest_tuned'), (gb_best, 'GradientBoosting_tuned')]:
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE={mae:.2f}, R2={r2:.3f}")
# %%
models = {
 'RidgeRegression_V': Ridge(alpha=1.0),
 'RandomForest_V': RandomForestRegressor(**gs_rf.best_params_, random_state=42),
 'GradientBoosting_V': GradientBoostingRegressor(**gs_gb.best_params_, random_state=42)
 }
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
# Display results
for name, metrics in results.items():
    print(f"{name}: MAE={metrics['MAE']:.2f}, MSE={metrics['MSE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")
#%%
#create pickle files
import pickle
for name, model_obj in models.items():
    # Construct a unique filename for each model using an f-string
    filename = f"{name}.pkl"
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model_obj, file) # Dump the actual model object, not the name
        print(f"Successfully pickled model '{name}' to {filename}")
    except Exception as e:
        print(f"Error pickling model '{name}': {e}")
# %%
#model comparison
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=preds, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted Prices (Gradient Boosting)")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()
# %%
