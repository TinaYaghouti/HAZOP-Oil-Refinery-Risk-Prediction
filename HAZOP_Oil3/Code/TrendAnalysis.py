# Trend Analysis & Moving Average

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data 

df = pd.read_csv('Data/HAZOP_DATA.csv')

# Add timestamp (simulated)

df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

# Moving average for temperature

# 5 hour moving average:

df['temp_ma5'] = df['temperature'].rolling(5).mean() 

# 10 hour moving average:

df['temp_ma10'] = df['temperature'].rolling(10).mean()

plt.figure(figsize=(15,6))
plt.plot(df['timestamp'], df['temperature'], label='Temperature', alpha=0.5)
plt.plot(df["timestamp"], df['temp_ma5'], label='MA 5h')
plt.plot(df["timestamp"], df['temp_ma10'], label='MA 10h')
plt.legend()
plt.title('Moving Average & Trend')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Plots/trend_analysis.png')
plt.close()

print("Trend analysis plot saved in 'Plots/'")
