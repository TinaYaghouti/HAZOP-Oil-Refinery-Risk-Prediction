from ConfigLoader import load_config, get_active_unit, get_features_for_unit

config = load_config()
unit_name, unit = get_active_unit(config)
feature_cols = get_features_for_unit(config, unit_name)

df = pd.read_csv('Data/operational_tehran.csv')
X = df[feature_cols]
y = df['risk']


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBRFClassifier
import pickle

# Load data

df = pd.read_csv('HAZOP_DATA.csv')

# Features

FeatureCols = ['temperature', 'pressure', 'flow_rate', 'level', 'vibration']

X = df[FeatureCols]
y = df['risk']

# Normalizition

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train & Test split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# XGBoost Model

model = XGBRFClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

# Train the model

model.fit(X_train, y_train)

# Predict on test data

y_pred = model.predict(X_test)

# Evaluation

print("Classification Report: ")
print(classification_report(y_test, y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

# Save model and scaler

with open('risk_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model saved as risk_model.pkl")
print("Scaler saved as scaler.pkl")