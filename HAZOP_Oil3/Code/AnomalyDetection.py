import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


os.makedirs("Plots", exist_ok=True)

# Load data 

df = pd.read_csv('Data/HAZOP_DATA.csv')
features = ['temperature', 'pressure', 'flow_rate', 'level', 'vibration']
X = df[features]

model = IsolationForest(contamination=0.1, random_state=42)
df['anomaly'] = model.fit_predict(X)

with open("Models/anomaly_model.pkl", "wb") as f:
    pickle.dump(model, f)

plt.figure(figsize=(10,6))
colors = df['anomaly'].map({1: 'blue', -1: 'red'})
plt.scatter(df['temperature'], df['pressure'], c=colors, alpha=0.6)
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.title('Anomaly Detection (Red = Anomaly)')
plt.savefig('Plots/anomaly_detection.png')
plt.close()

print("Anomaly model saved, plot saved in 'Plots/'")