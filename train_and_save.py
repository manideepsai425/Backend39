# ============================================================
#  train_and_save.py
#  Run this ONCE locally (or in Colab) to train models and
#  produce the .pkl files needed by the FastAPI backend.
#
#  Usage:
#    pip install -r requirements.txt
#    python train_and_save.py
# ============================================================

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

# ── Config ────────────────────────────────────────────────────
CSV_PATH   = os.path.join(os.path.dirname(__file__),
                          "data", "cleaned_food_supply_chain_sourcetable.csv")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(CSV_PATH)
print(f"  Shape: {df.shape}")

# ── Preprocessing ─────────────────────────────────────────────
df["date"]        = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek
df["month"]       = df["date"].dt.month
df["quarter"]     = df["date"].dt.quarter
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

le_crop   = LabelEncoder().fit(df["crop_name"])
le_region = LabelEncoder().fit(df["region"])
le_season = LabelEncoder().fit(df["season"])

df["crop_encoded"]   = le_crop.transform(df["crop_name"])
df["region_encoded"] = le_region.transform(df["region"])
df["season_encoded"] = le_season.transform(df["season"])

df = df.sort_values("date").reset_index(drop=True)
df["demand_lag_1"]    = df.groupby("crop_name")["demand_estimate"].shift(1).fillna(method="bfill")
df["demand_lag_7"]    = df.groupby("crop_name")["demand_estimate"].shift(7).fillna(method="bfill")
df["avg_7day_demand"] = (
    df.groupby("crop_name")["demand_estimate"]
      .transform(lambda x: x.rolling(7,  min_periods=1).mean())
)
df["avg_30day_demand"] = (
    df.groupby("crop_name")["demand_estimate"]
      .transform(lambda x: x.rolling(30, min_periods=1).mean())
)

# ── Feature Lists ─────────────────────────────────────────────
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

# ── Model 1: Demand (XGBoost) ─────────────────────────────────
print("\nTraining Demand model (XGBoost) …")
X_d = df[DEMAND_FEATURES]; y_d = df["demand_estimate"]
Xtr_d, Xte_d, ytr_d, yte_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)
demand_model = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbosity=0
)
demand_model.fit(Xtr_d, ytr_d, eval_set=[(Xte_d, yte_d)], verbose=False)
print("  Done ✅")

# ── Model 2: Spoilage (Random Forest) ────────────────────────
print("Training Spoilage model (Random Forest) …")
X_s = df[SPOILAGE_FEATURES]; y_s = df["spoilage_risk"]
Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
    X_s, y_s, test_size=0.2, random_state=42, stratify=y_s)
spoilage_model = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight="balanced", random_state=42, n_jobs=-1
)
spoilage_model.fit(Xtr_s, ytr_s)
print("  Done ✅")

# ── Model 3: Supply Gap (Gradient Boosting) ───────────────────
print("Training Supply Gap model (Gradient Boosting) …")
X_g = df[GAP_FEATURES]; y_g = df["supply_gap"]
Xtr_g, Xte_g, ytr_g, yte_g = train_test_split(X_g, y_g, test_size=0.2, random_state=42)
gap_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.08,
    max_depth=5, random_state=42
)
gap_model.fit(Xtr_g, ytr_g)
print("  Done ✅")

# ── Save All ──────────────────────────────────────────────────
print("\nSaving models …")
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

print("\n✅ All models saved to /models — ready for FastAPI!")
