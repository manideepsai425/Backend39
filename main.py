# ============================================================
# 🌾 Food Supply Chain AI — FastAPI Backend (FIXED)
# ============================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
import os

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
        raise RuntimeError(
            f"❌ Model file not found: {name}. "
            f"Ensure /models folder is uploaded with all .pkl files."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Lifespan (replaces deprecated @app.on_event) ─────────────
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
    yield  # App runs here
    # (cleanup on shutdown if needed goes after yield)

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="Food Supply Chain AI API",
    description="Predict demand, spoilage risk, and supply gap.",
    version="1.0.2",
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

# ── Request Schema ───────────────────────────────────────────
class PredictionRequest(BaseModel):
    crop: str
    region: str
    season: str
    month: int          = Field(..., ge=1, le=12)
    day_of_week: int    = Field(..., ge=0, le=6)
    is_weekend: int     = Field(..., ge=0, le=1)
    weather_temp: float
    rainfall_mm: float
    humidity_pct: float
    festival_flag: int  = Field(..., ge=0, le=1)
    holiday_flag: int   = Field(..., ge=0, le=1)
    price_per_quintal: float
    fuel_price: float
    transport_cost: float
    crop_yield: int
    shelf_life_days: int
    market_arrival: float
    demand_lag_1: float
    demand_lag_7: float
    avg_7day_demand: float
    avg_30day_demand: float

# ── Response Schema ──────────────────────────────────────────
class PredictionResponse(BaseModel):
    crop: str
    region: str
    season: str
    predicted_demand: float
    predicted_supply_gap: float
    spoilage_risk_label: str
    spoilage_risk_probability: float
    alert: str

# ── Helper ───────────────────────────────────────────────────
def safe_encode(encoder, value: str, field_name: str):
    classes = list(encoder.classes_)
    if value not in classes:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name}: {value}. Valid: {classes}"
        )
    return int(encoder.transform([value])[0])

# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "🌾 API running", "docs": "/docs"}

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
    crop_enc   = safe_encode(le_crop,   req.crop,   "crop")
    region_enc = safe_encode(le_region, req.region, "region")
    season_enc = safe_encode(le_season, req.season, "season")
    quarter    = (req.month - 1) // 3 + 1

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
