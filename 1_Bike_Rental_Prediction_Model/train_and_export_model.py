import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Define the features exactly as requested
features = ['temp', 'weathersit', 'mnth', 'hr', 'windspeed', 'workingday', 'weekday', 'yr', 'holiday']
target = 'cnt'

print("Loading data...")
df = pd.read_csv('BikeRentalData.csv')

print("Selecting features...")
X = df[features]
y = df[target]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

print("Training model...")
# Using parameters discovered in notebook: n_estimators=200, min_samples_split=5, random_state=42
model = RandomForestRegressor(n_estimators=200, min_samples_split=5, random_state=42)
model.fit(X_train_scaled, y_train)

print("Saving model and scaler...")
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
