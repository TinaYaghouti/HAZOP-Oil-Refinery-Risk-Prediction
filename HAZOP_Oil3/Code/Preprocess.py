from ConfigLoader import load_config, get_active_unit, get_features_for_unit
import pandas as pd

config = load_config()
unit_name, unit = get_active_unit(config)
feature_cols = get_features_for_unit(config, unit_name)

df = pd.read_csv('Data/HAZOP_DATA.csv')
X = df[feature_cols]
y = df['risk']


import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load data from phase 1 (GeneratedData.

df = pd.read_csv('HAZOP_DATA.csv')

# Feature Enginnering 

df['temp_pressure'] = df['temperature'] * df['pressure']
df['pressure_ratio'] = df['pressure'] / df['flow_rate']
df['level_deviation'] = np.abs(df['level'] - 50)

# Select features

FeatureCols = ['temperature', 'pressure', 'flow_rate', 'level', 'vibration', 'temp_pressure', 'pressure_ratio', 'level_deviation']

X = df[FeatureCols]
y = df['risk']

# Normalization

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train & Test split

X_train, X_test, y.train, y.test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save scaler for later use

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print('Preprocessing completed')
print(f"Number of features: {len(FeatureCols)}")
print(f"Training data size: {len(X_train)}")
print(f"Test data size: {len(X_test)}")
print("scaler.pkl Saved")