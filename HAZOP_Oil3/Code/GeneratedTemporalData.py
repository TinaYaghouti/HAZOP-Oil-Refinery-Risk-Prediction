import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 5000
time = np.arange(n)

temperature = 60 + 10 * np.sin(time / 50) + np.random.normal(0, 2, n)
pressure    = 100 + 15 * np.sin(time / 30 + 1) + np.random.normal(0, 3, n)
flow_rate   = 50 + 5 * np.sin(time / 20 + 2) + np.random.normal(0, 1, n)
level       = 40 + 8 * np.sin(time / 40 + 3) + np.random.normal(0, 1, n)
vibration   = 5 + 2 * np.sin(time / 15 + 4) + np.random.normal(0, 0.5, n)

risk = np.zeros(n, dtype=int)
for i in range(10, n):
    temp_trend = temperature[i] - temperature[i-5]
    press_trend = pressure[i] - pressure[i-5]

    score = (
        0.4 * (temperature[i] > 62) +
        0.3 * (pressure[i] > 100) +
        0.2 * (temp_trend > 0.5) +
        0.1 * (press_trend > 1.0) +
        0.1 * (vibration[i] > 5.5)
    )

    if score >= 0.5:
        risk[i] = 2
    elif score >= 0.25:
        risk[i] = 1
    else:
        risk[i] = 0

df = pd.DataFrame({
    'temperature': temperature,
    'pressure': pressure,
    'flow_rate': flow_rate,
    'level': level,
    'vibration': vibration,
    'risk': risk
})

print("Class distribution:")
print(df['risk'].value_counts())

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(base_dir, 'Data', 'HAZOP_DATA.csv')
df.to_csv(output_path, index=False)
print(f"\nData saved to: {output_path}")