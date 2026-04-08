# ============================================================
# 🌾 Food Supply Chain AI — FastAPI Backend (FIXED v2)
# ============================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ── Globals ──────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

demand_model   = None
spoilage_model = None
gap_model      = None
le_crop        = None
le_region      = None
le_season      = None

# ── Load Helper ──────────────────────────────────────────────
def load_pickle(name: str):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        raise RuntimeError(f"❌ Model file not found: {name}.")
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global demand_model, spoilage_model, gap_model
    global le_crop, le_region, le_season
    demand_model   = load_pickle("demand_model.pkl")
    spoilage_model = load_pickle("spoilage_model.pkl")
    gap_model      = load_pickle("gap_model.pkl")
    le_crop        = load_pickle("le_crop.pkl")
    le_region      = load_pickle("le_region.pkl")
    le_season      = load_pickle("le_season.pkl")
    yield

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="Food Supply Chain AI API",
    description="Predict demand, spoilage risk, and supply gap.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Feature columns ──────────────────────────────────────────
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

# ── Smart per-crop defaults (derived from real data) ─────────
CROP_DEFAULTS = {
    "Brinjal":  {"shelf_life_days": 5,   "price_per_quintal": 10056},
    "Cabbage":  {"shelf_life_days": 7,   "price_per_quintal": 12229},
    "Chilli":   {"shelf_life_days": 180, "price_per_quintal": 12567},
    "Maize":    {"shelf_life_days": 180, "price_per_quintal": 12734},
    "Onion":    {"shelf_life_days": 30,  "price_per_quintal": 11160},
    "Potato":   {"shelf_life_days": 60,  "price_per_quintal": 12959},
    "Rice":     {"shelf_life_days": 365, "price_per_quintal": 11931},
    "Tomato":   {"shelf_life_days": 5,   "price_per_quintal": 10805},
    "Turmeric": {"shelf_life_days": 365, "price_per_quintal": 11400},
    "Wheat":    {"shelf_life_days": 365, "price_per_quintal": 18438},
}

# ── Request Schema (all fields optional except core 3) ───────
class PredictionRequest(BaseModel):
    # ── Required core fields ──
    crop:   str
    region: str
    season: str

    # ── Environmental (required, sent by form) ──
    weather_temp: float
    rainfall_mm:  float

    # ── Optional — auto-computed or defaulted ──
    humidity_pct:      Optional[float] = 58.0
    month:             Optional[int]   = None   # auto from today
    day_of_week:       Optional[int]   = None   # auto from today
    is_weekend:        Optional[int]   = None   # auto from today
    festival_flag:     Optional[int]   = 0
    holiday_flag:      Optional[int]   = 0
    price_per_quintal: Optional[float] = None   # auto from crop
    fuel_price:        Optional[float] = 98.6
    transport_cost:    Optional[float] = 409.6
    crop_yield:        Optional[int]   = 3878
    shelf_life_days:   Optional[int]   = None   # auto from crop
    market_arrival:    Optional[float] = 136.8
    demand_lag_1:      Optional[float] = 1342.0
    demand_lag_7:      Optional[float] = 1342.0
    avg_7day_demand:   Optional[float] = 1342.0
    avg_30day_demand:  Optional[float] = 1342.0

# ── Response Schema ──────────────────────────────────────────
class PredictionResponse(BaseModel):
    crop:                      str
    region:                    str
    season:                    str
    predicted_demand:          float
    predicted_supply_gap:      float
    spoilage_risk_label:       str
    spoilage_risk_probability: float
    alert:                     str

# ── Helper ───────────────────────────────────────────────────
def safe_encode(encoder, value: str, field_name: str):
    classes = list(encoder.classes_)
    if value not in classes:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name}: '{value}'. Valid values: {classes}"
        )
    return int(encoder.transform([value])[0])

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "🌾 Food Supply Chain AI API is running.",
        "docs":    "/docs",
        "predict": "/predict"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/options")
def get_options():
    return {
        "crops":   list(le_crop.classes_),
        "regions": list(le_region.classes_),
        "seasons": list(le_season.classes_)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # ── Encode categorical fields ────────────────────────────
    crop_enc   = safe_encode(le_crop,   req.crop,   "crop")
    region_enc = safe_encode(le_region, req.region, "region")
    season_enc = safe_encode(le_season, req.season, "season")

    # ── Auto-fill date fields ────────────────────────────────
    now         = datetime.now()
    month       = req.month       if req.month       is not None else now.month
    day_of_week = req.day_of_week if req.day_of_week is not None else now.weekday()
    is_weekend  = req.is_weekend  if req.is_weekend  is not None else int(now.weekday() >= 5)
    quarter     = (month - 1) // 3 + 1

    # ── Auto-fill crop-specific fields ───────────────────────
    crop_def          = CROP_DEFAULTS.get(req.crop, {})
    shelf_life_days   = req.shelf_life_days   if req.shelf_life_days   is not None else crop_def.get("shelf_life_days",   30)
    price_per_quintal = req.price_per_quintal if req.price_per_quintal is not None else crop_def.get("price_per_quintal", 12000)

    # ── Resolve remaining fields ─────────────────────────────
    humidity_pct   = req.humidity_pct
    festival_flag  = req.festival_flag
    holiday_flag   = req.holiday_flag
    fuel_price     = req.fuel_price
    transport_cost = req.transport_cost
    crop_yield     = req.crop_yield
    market_arrival = req.market_arrival
    demand_lag_1   = req.demand_lag_1
    demand_lag_7   = req.demand_lag_7
    avg_7day       = req.avg_7day_demand
    avg_30day      = req.avg_30day_demand

    # ── Build feature DataFrames ─────────────────────────────
    demand_input = pd.DataFrame([[
        day_of_week, month, quarter, is_weekend,
        crop_enc, region_enc, season_enc,
        shelf_life_days,
        price_per_quintal, req.weather_temp, req.rainfall_mm, humidity_pct,
        festival_flag, holiday_flag,
        fuel_price, transport_cost, crop_yield,
        demand_lag_1, demand_lag_7, avg_7day, avg_30day
    ]], columns=DEMAND_FEATURES)

    spoi_input = pd.DataFrame([[
        day_of_week, month, is_weekend,
        crop_enc, region_enc, season_enc,
        shelf_life_days,
        req.weather_temp, req.rainfall_mm, humidity_pct,
        market_arrival, transport_cost,
        festival_flag, holiday_flag,
        demand_lag_1
    ]], columns=SPOILAGE_FEATURES)

    gap_input = pd.DataFrame([[
        day_of_week, month, quarter,
        crop_enc, region_enc, season_enc,
        market_arrival, demand_lag_1,
        price_per_quintal, req.weather_temp,
        festival_flag, holiday_flag,
        crop_yield, transport_cost, fuel_price
    ]], columns=GAP_FEATURES)

    # ── Run predictions ──────────────────────────────────────
    pred_demand = float(demand_model.predict(demand_input)[0])
    pred_risk   = int(spoilage_model.predict(spoi_input)[0])
    pred_risk_p = float(spoilage_model.predict_proba(spoi_input)[0][1]) if hasattr(spoilage_model, "predict_proba") else 0.0
    pred_gap    = float(gap_model.predict(gap_input)[0])

    alert = (
        f"⚠️ Shortage of {abs(pred_gap):.0f} units expected!"
        if pred_gap < 0
        else f"✅ Supply OK. Surplus: {pred_gap:.0f} units"
    )

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
