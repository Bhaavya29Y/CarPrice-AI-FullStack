# ============================================================
#   BACKEND SERVER  |  backend/main.py
#   Run: uvicorn main:app --reload
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI(title="CarPrice AI Backend API")

# ── Load your Colab exported assets ─────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# ── Define the incoming Data Schema ──────────────────────────
class CarFeatures(BaseModel):
    brand: str
    year: int
    km_driven: int
    fuel: str
    transmission: str
    owner: str
    seller_type: str
    mileage: float
    engine: int
    max_power: float
    seats: int

# ── Helper function to safely map labels to numbers ──────────
def encode_val(col_name, user_input):
    # Check if this column has a saved LabelEncoder from your Colab training
    if col_name in encoders:
        try:
            # Safely handle unseen inputs or map the strings back to numeric codes
            return encoders[col_name].transform([str(user_input)])[0]
        except Exception:
            # Fallback to 0 if something unexpected is passed
            return 0
    return user_input

# ── The API Endpoint your Frontend will talk to ──────────────
@app.post("/predict")
def predict_car_price(data: CarFeatures):
    # Map the incoming stream inputs to the column names your model trained on
    input_map = {
        "name":         encode_val("name", data.brand),
        "brand":        encode_val("brand", data.brand),
        "year":         data.year,
        "km_driven":    data.km_driven,
        "fuel":         encode_val("fuel", data.fuel),
        "fuel_type":    encode_val("fuel_type", data.fuel),
        "transmission": encode_val("transmission", data.transmission),
        "owner":        encode_val("owner", data.owner),
        "seller_type":  encode_val("seller_type", data.seller_type),
        "mileage":      data.mileage,
        "engine":       data.engine,
        "max_power":    data.max_power,
        "seats":        data.seats,
    }
    
    # Sort data into the precise array structural order your model expects
    row = [input_map.get(col, 0) for col in feature_cols]
    
    # Scale features and inverse-log the exponential target output prediction
    row_sc = scaler.transform([row])
    log_pred = model.predict(row_sc)[0]
    price = np.expm1(log_pred)
    
    return {"predicted_price": int(round(price))}