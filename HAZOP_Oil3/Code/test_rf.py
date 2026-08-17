import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, 'Data', 'HAZOP_DATA.csv'))

features = ['temperature', 'pressure', 'flow_rate', 'level', 'vibration']
X = df[features].values
y = df['risk'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
split = int(len(X_scaled) * 0.8)
X_train_rf = X_scaled[:split]
X_test_rf = X_scaled[split:]
y_train_rf = y[:split]
y_test_rf = y[split:]

# Train
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_rf, y_train_rf)
print("Random Forest Accuracy:", accuracy_score(y_test_rf, rf.predict(X_test_rf)))

print("\nFeature importance:")
for f, i in zip(features, rf.feature_importances_):
    print(f"{f}: {i:.3f}")