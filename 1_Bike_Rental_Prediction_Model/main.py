from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="Bike Rental Prediction API")

# Load the model and scaler during startup
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    model = None
    scaler = None
    print(f"Warning: Could not load model or scaler. Is train_and_export_model.py executed? Error: {e}")

# Define the Pydantic model for input data validation
class BikeRentalInput(BaseModel):
    # 'temp', 'weathersit', 'mnth', 'hr', 'windspeed', 'workingday', 'weekday', 'yr', 'holiday'
    temp: float = Field(..., description="Normalized temperature in Celsius. Values derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)")
    weathersit: int = Field(..., description="Weather Situation: 1: Clear, 2: Mist, 3: Light Snow/Rain, 4: Heavy Rain/Snow")
    mnth: int = Field(..., ge=1, le=12, description="Month (1 to 12)")
    hr: int = Field(..., ge=0, le=23, description="Hour (0 to 23)")
    windspeed: float = Field(..., description="Normalized wind speed")
    workingday: int = Field(..., description="If day is neither weekend nor holiday is 1, otherwise is 0")
    weekday: int = Field(..., ge=0, le=6, description="Day of the week (0-6)")
    yr: int = Field(..., description="Year (0: 2011, 1: 2012)")
    holiday: int = Field(..., description="Whether day is holiday or not (0, 1)")

class PredictionOutput(BaseModel):
    predicted_rentals: float
    input_data: dict

@app.post("/predict", response_model=PredictionOutput)
async def predict_rentals(data: BikeRentalInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded properly.")
    
    # Convert input to DataFrame matching the scaler's expected format
    input_df = pd.DataFrame([data.model_dump()])
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    # Ensure prediction is non-negative and round to integer
    predicted_rentals = max(0, round(prediction))
    
    return PredictionOutput(
        predicted_rentals=predicted_rentals,
        input_data=data.model_dump()
    )

# Mount static folder for serving frontend
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

