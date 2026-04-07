🌾 Food Supply Chain AI — FastAPI Backend
AI-powered prediction API for agricultural supply chain optimization.
Predicts demand, spoilage risk, and supply gap for crops across South Indian regions.
Models
Model
Algorithm
Target
Demand Forecast
XGBoost Regressor
demand_estimate
Spoilage Risk
Random Forest Classifier
spoilage_risk (0/1)
Supply Gap
Gradient Boosting Regressor
supply_gap
Project Structure
food-supply-chain-backend/
├── main.py                  ← FastAPI app (all endpoints)
├── train_and_save.py        ← Run once to train & save models
├── requirements.txt
├── render.yaml              ← Render deployment config
├── data/
│   └── cleaned_food_supply_chain_sourcetable.csv
└── models/                  ← Generated after running train_and_save.py
    ├── demand_model.pkl
    ├── spoilage_model.pkl
    ├── gap_model.pkl
    ├── le_crop.pkl
    ├── le_region.pkl
    └── le_season.pkl
Local Setup
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (only needed once)
python train_and_save.py

# 3. Start server
uvicorn main:app --reload
API docs available at: http://localhost:8000/docs
Endpoints
Method
Path
Description
GET
/
Health check + info
GET
/health
Simple OK response
GET
/options
Valid crop/region/season values
POST
/predict
Run all 3 model predictions
Deploy on Render
Push this repository to GitHub.
Go to render.com → New Web Service.
Connect your GitHub repo.
Render auto-detects render.yaml — click Deploy.
Your API will be live at https://<your-service>.onrender.com.
Important: The models/ folder must be committed with the .pkl files.
Run train_and_save.py locally first, then git add models/ before pushing.
Example Request
curl -X POST https://<your-service>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "Rice",
    "region": "Telangana",
    "season": "Kharif",
    "month": 8,
    "day_of_week": 2,
    "is_weekend": 0,
    "weather_temp": 32.0,
    "rainfall_mm": 45.0,
    "humidity_pct": 68.0,
    "festival_flag": 0,
    "holiday_flag": 0,
    "price_per_quintal": 12000.0,
    "fuel_price": 100.0,
    "transport_cost": 400.0,
    "crop_yield": 2500,
    "shelf_life_days": 30,
    "market_arrival": 500.0,
    "demand_lag_1": 1200.0,
    "demand_lag_7": 1100.0,
    "avg_7day_demand": 1150.0,
    "avg_30day_demand": 1080.0
  }'
Example Response
{
  "crop": "Rice",
  "region": "Telangana",
  "season": "Kharif",
  "predicted_demand": 1243.5,
  "predicted_supply_gap": 87.2,
  "spoilage_risk_label": "LOW RISK",
  "spoilage_risk_probability": 18.4,
  "alert": "✅ Supply OK. Surplus: 87 units"
}