from ConfigLoader import load_config, get_hazop_rules

config = load_config()
hazop_rules = get_hazop_rules(config)

def explain_risk(row):
    reasons = []
    if row['temperature'] > 100: reasons.append("High temp")
    return " - ".join(reasons) if reasons else "Normal"



import numpy as np
import pandas as pd
from sklearn.preprocessing import  StandardScaler
from xgboost import XGBRFClassifier
import pickle

# Load or generate data

try:
    df = pd.read_csv('HAZOP_DATA.csv')
except:
    print("Generating fresh data...")
    np.random.seed(42)
    n = 5000
    data = {
        'temperature': np.random.uniform(50, 150, n),
        'pressure': np.random.uniform(5, 20, n),
        'flow_rate': np.random.uniform(100, 500, n),
        'level': np.random.uniform(0, 100, n),
        'vibration': np.random.uniform(0, 10, n)
 } 
    
    df = pd.DataFrame(data)

def HAZOP_RISK(row):
    score = 0
    if row['temperature'] > 100: score += 1
    if row['pressure'] > 15: score += 1
    if row['vibration'] > 8: score += 1
    if row['flow_rate'] > 450: score += 1
    if row['level'] > 85 or row['level'] < 15: score += 1
    if score >= 3: return 2
    elif score >= 1: return 1
    else: return 0
df['risk'] = df.apply(HAZOP_RISK, axis=1)

# Train model

FeatuteCols = ['temperature', 'pressure', 'flow_rate', 'level', 'vibration']

X = df[FeatuteCols]
y = df['risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBRFClassifier(n_estimators=100, max_depth=6, random_state=42)

model.fit(X_scaled, y)

# Save model and scaler

with open('risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved")

# Root cause analysis function

def ExplainRisk(row):
    reasons = []
    if row['temperature'] > 100:
        reasons.append("High temperature")
    if row['pressure'] > 15:
        reasons.append("High pressure")
    if row['vibration'] > 8:
        reasons.append("High vibration")
    if row['flow_rate'] < 120:
        reasons.append("Low flow rate")
    elif row['flow_rate'] > 450:
        reasons.append("High flow rate")
    if row['level'] > 85:
        reasons.append("High level")
    elif row['level'] < 15:
        reasons.append("Low level")

    if not reasons:
        return "Normal (no specific factor)"
    return " - ".join(reasons)

# Add explanation column

df['risk_reason'] = df.apply(ExplainRisk, axis=1)

# Show samples

print("\n 5 samples of output (risk + reasons): ")
print(df[['temperature', 'pressure', 'vibration', 'risk', 'risk_reason']].head())

# Save final dataset

df.to_csv('HAZOP_DATA_with_reasons.csv', index=False)
print("\n Final dataset saved: HAZOP_DATA_with_reasons.csv")