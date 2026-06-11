import numpy as np
import pickle
import json

# Model Utility Functions: 
# this file provides helper functions for the API team to easily load the model, make predictions, and get root cause explanations
# without rewriting the same code

import pickle
import numpy as np
import json

# PHASE 1

def load_model_and_scaler(model_path='Models/risk_model.pkl', scaler_path='Models/scaler.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_risk(model, scaler, temp, press, flow, level, vib):
    feat = np.array([[temp, press, flow, level, vib]])
    feat_scaled = scaler.transform(feat)
    pred = model.predict(feat_scaled)[0]
    return {0:"Low",1:"Medium",2:"High"}[pred]


# PHASE 2

def load_config(path="risk_config.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def predict_risk_by_category(category, features_dict):
    config = load_config()
    unit = list(config["refinery_units"].values())[0]
    feature_order = unit["risk_categories"][category]["features"]
    features = np.array([[features_dict[f] for f in feature_order]])

    if category == "operational":
        model, scaler = load_model_and_scaler()
    else:
        with open(f"Models/OtherRisks/{category}_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"Models/OtherRisks/{category}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    return {0:"Low",1:"Medium",2:"High"}[pred]

def explain_risk_enhanced(temp, press, vib, flow=None, level=None):
    reasons = []
    if temp > 100: reasons.append("High temp")
    if press > 15: reasons.append("High press")
    if vib > 8: reasons.append("High vib")
    if temp > 100 and press > 15:
        reasons.append("Operator fatigue possible")
    if vib > 8 and temp > 90:
        reasons.append("Maintenance delay")
    return " - ".join(reasons) if reasons else "Normal"