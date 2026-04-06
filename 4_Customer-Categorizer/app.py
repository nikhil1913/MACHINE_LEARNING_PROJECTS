from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Customer Categorizer API", description="API to predict customer cluster")

# Load model and feature names
model_data = joblib.load("model.pkl")
model = model_data["model"]
features = model_data["features"]

# Determine current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

class PredictionRequest(BaseModel):
    Age: int
    Education: int
    Marital_Status: int # Will map to Marital Status
    Parental_Status: int # Will map to Parental Status
    Children: int
    Income: float
    Total_Spending: int
    Days_as_Customer: int
    Recency: int
    Wines: int
    Fruits: int
    Meat: int
    Fish: float
    Sweets: int
    Gold: float
    Web: int
    Catalog: int
    Store: float
    Discount_Purchases: int # Will map to Discount Purchases
    Total_Promo: int # Will map to Total Promo
    NumWebVisitsMonth: int

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r") as f:
        return f.read()

@app.post("/predict")
async def predict(data: PredictionRequest):
    # Convert incoming data to a DataFrame corresponding exactly to trained features
    # Map the pydantic model fields (which cannot have spaces) back to spaces
    mapping = {
        "Marital_Status": "Marital Status",
        "Parental_Status": "Parental Status",
        "Discount_Purchases": "Discount Purchases",
        "Total_Promo": "Total Promo"
    }
    
    input_dict = data.dict()
    mapped_dict = {}
    for k, v in input_dict.items():
        if k in mapping:
            mapped_dict[mapping[k]] = v
        else:
            mapped_dict[k] = v
            
    df = pd.DataFrame([mapped_dict])
    
    # Ensure columns match training
    df = df[features]
    
    # Predict
    cluster = model.predict(df)[0]
    
    CLUSTER_NAMES = {
        0: "Value-Conscious Parents",
        1: "Budget Shoppers",
        2: "Premium Customers"
    }
    
    return {
        "cluster": int(cluster),
        "label": CLUSTER_NAMES.get(int(cluster), "Unknown")
    }
