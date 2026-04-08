import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "cleaned_fixed.csv")   # FIX: corrected filename
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

print("Columns:", df.columns.tolist())

# ── Preprocessing ──────────────────────────────────────
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

# Fill missing values
df = df.ffill().bfill()

# Feature engineering
df["day_of_week"] = df["date"].dt.dayofweek
df["month"]       = df["date"].dt.month
df["quarter"]     = df["date"].dt.quarter
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# Encoding
df["crop_name"] = df["crop_name"].astype(str)
df["region"]    = df["region"].astype(str)
df["season"]    = df["season"].astype(str)

le_crop   = LabelEncoder().fit(df["crop_name"])
le_region = LabelEncoder().fit(df["region"])
le_season = LabelEncoder().fit(df["season"])

df["crop_encoded"]   = le_crop.transform(df["crop_name"])
df["region_encoded"] = le_region.transform(df["region"])
df["season_encoded"] = le_season.transform(df["season"])

# Lag features — uses .bfill() (pandas 2.x compatible)
df = df.sort_values("date")
df["demand_lag_1"]    = df.groupby("crop_name")["demand_estimate"].shift(1).bfill()
df["demand_lag_7"]    = df.groupby("crop_name")["demand_estimate"].shift(7).bfill()
df["avg_7day_demand"] = df.groupby("crop_name")["demand_estimate"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df["avg_30day_demand"] = df.groupby("crop_name")["demand_estimate"].transform(
    lambda x: x.rolling(30, min_periods=1).mean()
)

df = df.ffill().bfill().dropna()
print("Preprocessing done ✅")

# ── Features ───────────────────────────────────────────
DEMAND_FEATURES = [
    "day_of_week", "month", "quarter", "is_weekend",
    "crop_encoded", "region_encoded", "season_encoded",
    "shelf_life_days",
    "price_per_quintal", "weather_temp", "rainfall_mm", "humidity_pct",
    "festival_flag", "holiday_flag",
    "fuel_price", "transport_cost", "crop_yield",
    "demand_lag_1", "demand_lag_7", "avg_7day_demand", "avg_30day_demand"
]

SPOILAGE_FEATURES = [
    "day_of_week", "month", "is_weekend",
    "crop_encoded", "region_encoded", "season_encoded",
    "shelf_life_days",
    "weather_temp", "rainfall_mm", "humidity_pct",
    "market_arrival", "transport_cost",
    "festival_flag", "holiday_flag",
    "demand_lag_1"
]

GAP_FEATURES = [
    "day_of_week", "month", "quarter",
    "crop_encoded", "region_encoded", "season_encoded",
    "market_arrival", "demand_lag_1",
    "price_per_quintal", "weather_temp",
    "festival_flag", "holiday_flag",
    "crop_yield", "transport_cost", "fuel_price"
]

# ── Validate all feature columns exist ─────────────────  # FIX: added validation
all_features = set(DEMAND_FEATURES + SPOILAGE_FEATURES + GAP_FEATURES)
missing = all_features - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# ── Train Models ───────────────────────────────────────
print("Training models...")

X_d, y_d = df[DEMAND_FEATURES], df["demand_estimate"]
X_s, y_s = df[SPOILAGE_FEATURES], df["spoilage_risk"]
X_g, y_g = df[GAP_FEATURES], df["supply_gap"]

demand_model   = xgb.XGBRegressor(n_estimators=200, verbosity=0).fit(X_d, y_d)
spoilage_model = RandomForestClassifier(n_estimators=150, random_state=42).fit(X_s, y_s)
gap_model      = GradientBoostingRegressor(random_state=42).fit(X_g, y_g)

print("Models trained ✅")

# ── Save Models ────────────────────────────────────────
print("Saving models...")

objects = {
    "demand_model.pkl":   demand_model,
    "spoilage_model.pkl": spoilage_model,
    "gap_model.pkl":      gap_model,
    "le_crop.pkl":        le_crop,
    "le_region.pkl":      le_region,
    "le_season.pkl":      le_season,
}
for fname, obj in objects.items():
    with open(os.path.join(MODEL_DIR, fname), "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → models/{fname}")

print("✅ All models saved successfully!")
