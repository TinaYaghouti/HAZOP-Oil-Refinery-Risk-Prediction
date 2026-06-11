import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import os

os.makedirs("Models/OTHER_RISKS", exist_ok=True)

# Strategic Risk Model

df = pd.read_csv("Data/OTHER_RISKS/STRATEGIC_RISK.csv")
features = ["exchange_rate", "oil_price", "sanctions_index"]
X = df[features]
y = df["risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

with open("Models/OTHER_RISKS/strategic_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("Models/OTHER_RISKS/strategic_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Strategic model trained")
print(classification_report(y_test, model.predict(X_test)))

# Financial Risk Model

df = pd.read_csv("Data/OTHER_RISKS/FINANCIAL_RISK.csv")
features = ["usd_irr", "inflation_rate", "interest_rate"]
X = df[features]
y = df["risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

with open("Models/OTHER_RISKS/financial_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("Models/OTHER_RISKS/financial_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Financial model trained")
print(classification_report(y_test, model.predict(X_test)))

# Legal Risk Model

df = pd.read_csv("Data/OTHER_RISKS/LEGAL_RISK.csv")
features = ["audit_score", "contract_deviations", "compliance_violations"]
X = df[features]
y = df["risk"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

with open("Models/OTHER_RISKS/legal_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("Models/OTHER_RISKS/legal_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n Legal model trained")
print(classification_report(y_test, model.predict(X_test)))