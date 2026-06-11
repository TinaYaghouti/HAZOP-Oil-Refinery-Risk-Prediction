import pickle
import numpy as np
import json

def load_model_scaler(cat):
    if cat == "operational":
        with open("Models/risk_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("Models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    else:
        with open(f"Models/OTHER_RISKS/{cat}_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"Models/OTHER_RISKS/{cat}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    return model, scaler

def predict_risk(category, features_dict):
    with open("risk_config.json", "r") as f:
        config = json.load(f)
    unit = list(config["refinery_units"].values())[0]
    feature_order = unit["risk_categories"][category]["features"]
    features = np.array([[features_dict[f] for f in feature_order]])
    model, scaler = load_model_scaler(category)
    scaled = scaler.transform(features)
    pred = model.predict(scaled)[0]
    return {0:"Low",1:"Medium",2:"High"}[pred]

# test

if __name__ == "__main__":
    print(predict_risk("strategic", {"exchange_rate":52000, "oil_price":85, "sanctions_index":60}))