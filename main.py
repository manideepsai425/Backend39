# ============================================================
#  🌾 Food Supply Chain AI — FastAPI Backend
#  Wraps Demand, Spoilage Risk, and Supply Gap models
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
import os

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="Food Supply Chain AI API",
    description="Predict demand, spoilage risk, and supply gap for agricultural supply chains.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models ───────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_pickle(name: str):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}. Run train_and_save.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)

demand_model   = load_pickle("demand_model.pkl")
spoilage_model = load_pickle("spoilage_model.pkl")
gap_model      = load_pickle("gap_model.pkl")
le_crop        = load_pickle("le_crop.pkl")
le_region      = load_pickle("le_region.pkl")
le_season      = load_pickle("le_season.pkl")

# ── Feature column lists (must match training order) ─────────
DEMAND_FEATURES = [
    'day_of_week', 'month', 'quarter', 'is_weekend',
    'crop_encoded', 'region_encoded', 'season_encoded',
    'shelf_life_days',
    'price_per_quintal', 'weather_temp', 'rainfall_mm', 'humidity_pct',
    'festival_flag', 'holiday_flag',
    'fuel_price', 'transport_cost', 'crop_yield',
    'demand_lag_1', 'demand_lag_7', 'avg_7day_demand', 'avg_30day_demand'
]

SPOILAGE_FEATURES = [
    'day_of_week', 'month', 'is_weekend',
    'crop_encoded', 'region_encoded', 'season_encoded',
    'shelf_life_days',
    'weather_temp', 'rainfall_mm', 'humidity_pct',
    'market_arrival', 'transport_cost',
    'festival_flag', 'holiday_flag',
    'demand_lag_1'
]

GAP_FEATURES = [
    'day_of_week', 'month', 'quarter',
    'crop_encoded', 'region_encoded', 'season_encoded',
    'market_arrival', 'demand_lag_1',
    'price_per_quintal', 'weather_temp',
    'festival_flag', 'holiday_flag',
    'crop_yield', 'transport_cost', 'fuel_price'
]

# ── Request Schema ────────────────────────────────────────────
class PredictionRequest(BaseModel):
    crop: str           = Field(..., example="Rice",
                                description="One of: Rice, Potato, Turmeric, Tomato, Brinjal, Cabbage, Onion, Chilli, Maize, Wheat")
    region: str         = Field(..., example="Telangana",
                                description="One of: Andhra Pradesh, Telangana, Maharashtra, Tamil Nadu, Karnataka")
    season: str         = Field(..., example="Kharif",
                                description="One of: Summer, Kharif, Rabi")
    month: int          = Field(..., ge=1, le=12, example=8)
    day_of_week: int    = Field(..., ge=0, le=6, example=2,
                                description="0=Monday … 6=Sunday")
    is_weekend: int     = Field(..., ge=0, le=1, example=0)
    weather_temp: float = Field(..., example=32.0)
    rainfall_mm: float  = Field(..., example=45.0)
    humidity_pct: float = Field(..., example=68.0)
    festival_flag: int  = Field(..., ge=0, le=1, example=0)
    holiday_flag: int   = Field(..., ge=0, le=1, example=0)
    price_per_quintal: float = Field(..., example=12000.0)
    fuel_price: float   = Field(..., example=100.0)
    transport_cost: float = Field(..., example=400.0)
    crop_yield: int     = Field(..., example=2500)
    shelf_life_days: int = Field(..., example=30)
    market_arrival: float = Field(..., example=500.0,
                                  description="Market arrival quantity (supply proxy)")
    demand_lag_1: float = Field(..., example=1200.0,
                                description="Previous day's demand")
    demand_lag_7: float = Field(..., example=1100.0,
                                description="Demand 7 days ago")
    avg_7day_demand: float  = Field(..., example=1150.0)
    avg_30day_demand: float = Field(..., example=1080.0)


# ── Response Schema ───────────────────────────────────────────
class PredictionResponse(BaseModel):
    crop: str
    region: str
    season: str
    predicted_demand: float
    predicted_supply_gap: float
    spoilage_risk_label: str       # "HIGH RISK" or "LOW RISK"
    spoilage_risk_probability: float
    alert: str


# ── Helper ────────────────────────────────────────────────────
def safe_encode(encoder, value: str, field_name: str):
    classes = list(encoder.classes_)
    if value not in classes:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name} '{value}'. Valid options: {classes}"
        )
    return int(encoder.transform([value])[0])


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "🌾 Food Supply Chain AI API is running.",
        "docs": "/docs",
        "predict": "/predict"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def get_options():
    """Return valid categorical options for the predict form."""
    return {
        "crops":   list(le_crop.classes_),
        "regions": list(le_region.classes_),
        "seasons": list(le_season.classes_),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # Encode categoricals
    crop_enc   = safe_encode(le_crop,   req.crop,   "crop")
    region_enc = safe_encode(le_region, req.region, "region")
    season_enc = safe_encode(le_season, req.season, "season")
    quarter    = (req.month - 1) // 3 + 1

    # Build DataFrames
    demand_input = pd.DataFrame([[
        req.day_of_week, req.month, quarter, req.is_weekend,
        crop_enc, region_enc, season_enc,
        req.shelf_life_days,
        req.price_per_quintal, req.weather_temp, req.rainfall_mm, req.humidity_pct,
        req.festival_flag, req.holiday_flag,
        req.fuel_price, req.transport_cost, req.crop_yield,
        req.demand_lag_1, req.demand_lag_7, req.avg_7day_demand, req.avg_30day_demand
    ]], columns=DEMAND_FEATURES)

    spoi_input = pd.DataFrame([[
        req.day_of_week, req.month, req.is_weekend,
        crop_enc, region_enc, season_enc,
        req.shelf_life_days,
        req.weather_temp, req.rainfall_mm, req.humidity_pct,
        req.market_arrival, req.transport_cost,
        req.festival_flag, req.holiday_flag,
        req.demand_lag_1
    ]], columns=SPOILAGE_FEATURES)

    gap_input = pd.DataFrame([[
        req.day_of_week, req.month, quarter,
        crop_enc, region_enc, season_enc,
        req.market_arrival, req.demand_lag_1,
        req.price_per_quintal, req.weather_temp,
        req.festival_flag, req.holiday_flag,
        req.crop_yield, req.transport_cost, req.fuel_price
    ]], columns=GAP_FEATURES)

    # Predictions
    pred_demand = float(demand_model.predict(demand_input)[0])
    pred_risk   = int(spoilage_model.predict(spoi_input)[0])
    pred_risk_p = float(spoilage_model.predict_proba(spoi_input)[0][1])
    pred_gap    = float(gap_model.predict(gap_input)[0])

    # Alert message
    if pred_gap < 0:
        alert = f"⚠️ Shortage of {abs(pred_gap):.0f} units expected!"
    else:
        alert = f"✅ Supply OK. Surplus: {pred_gap:.0f} units"

    return PredictionResponse(
        crop=req.crop,
        region=req.region,
        season=req.season,
        predicted_demand=round(pred_demand, 2),
        predicted_supply_gap=round(pred_gap, 2),
        spoilage_risk_label="HIGH RISK" if pred_risk == 1 else "LOW RISK",
        spoilage_risk_probability=round(pred_risk_p * 100, 2),
        alert=alert,
    )
