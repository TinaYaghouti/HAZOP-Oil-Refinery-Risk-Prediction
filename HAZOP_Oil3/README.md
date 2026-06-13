# HAZOP Risk Prediction For Oil Refinery

## Project Overview
Predicts risk level (Low/Medium/High) using XGBoost based on HAZOP standards & provides root cause analysis

## Features (order is import for API)
1. temperature
2. pressure
3. flow_rate
4. level
5. vibration

## Output

** risk **: 0=Low, 1=Medium, 2=High
** reasons **: Root cause explanation (e.g., "High temperature - High pressure)

## How To Use The Model In API

import pickle
import numpy as np

# Load Model & Scaler

with open('risk_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input (exact order)

sample = np.array([[95, 13, 300, 50, 5]])
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)[0]   #0,1,2





# Root Cause Function (for API team)

def ExplainRisk(temperature, pressure, vibration)
    reasons = []
    if temperature > 100:
    reasons.append("High temperature")
    if pressure > 15:
    reasons.append("High pressure")
    if vibration > 8:
    reasons.append("High vibration")
    return " - ".join(reasons) if reasons
else "Normal"

## Files In This Package

HAZOP_DATA.csv -> raw data

HAZOP_DATA_with_reasons.csv -> data with root cause column

risk_model.pkl -> trained XGBoost model

scaler.pkl -> StandardScaler for normalization

GeneratedData.py -> data generation 

Preprocess.py -> preprocessing & feature engineering

TrainModel.py -> model training 

RootCauseAnalysis.py -> root cause analysis

HAZOP_RISK_PREDICTION.ipynb -> complete notebook 

README.md -> this file 


# Model Performance 

Algorithm: XGBoost

Accuracy: ~93%

Risk Levels: Low(0), Medium(1), High(2)

# PHASE 2

# HAZOP Risk Prediction System – Scalable Multi-Risk Architecture

## Project Overview
This project predicts risk levels (Low/Medium/High) for different refinery units and risk categories (Operational, Strategic, Financial, Legal) using XGBoost, Isolation Forest, and LSTM.

## Scalability & Multi-Risk Support
- **Risk categories** are defined in `risk_config.json`
- **Each category** has its own features, subcategories, and data source
- **New refinery unit** can be added by simply adding a new entry in `risk_config.json` – no code change required
- **Data sources** support CSV, Excel, API, Database

## How to Add a New Refinery Unit
1. Open `risk_config.json`
2. Add a new unit under `refinery_units`
3. Define its `risk_categories`, `features`, and `data_source`
4. Run the pipeline – everything works automatically

# Model Performance

Operational XGBoost: ~93% accuracy

Strategic / Financial / Legal XGBoost: ~88-92% accuracy

LSTM: ~87% accuracy

# Contact

For any questions, feel free to ask