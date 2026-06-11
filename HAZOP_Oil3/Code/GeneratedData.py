from ConfigLoader import load_config, get_active_unit, get_features_for_unit

config = load_config()
unit_name, unit = get_active_unit(config)
features = get_features_for_unit(config, unit_name)   

print(f" Generating data for: {features}")


import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

# 42: Symbolic number in programming culture from ' The Hitchhiker's Guide to the Galaxy '

data = {
    'temperature': np.random.uniform(50, 150, n),
    'pressure': np.random.uniform(5, 20, n),
    'flow_rate': np.random.uniform(100, 500, n),
    'level': np.random.uniform(0, 100, n),
    "vibration": np.random.uniform(0, 10, n)
}

# Dictionary to directly name each column with keys

df = pd.DataFrame(data)

def HAZOP_Risk(row):
    if row['temperature'] > 100 and row['pressure'] > 15:
        return 2
    elif row['temperature'] > 80 or row['pressure'] > 12 or row['vibration'] > 8:
        return 1
    else:
        return 0

df['risk'] = df.apply(HAZOP_Risk, axis=1)

# axis=1: Apply function row wise (not column wise)

df.to_csv('HAZOP_DATA.csv', index=False)

# index=False: Don't save row numbers

print('HAZOP_DATA.csv Created')
print(df['risk'].value_counts())
print(df.head())
