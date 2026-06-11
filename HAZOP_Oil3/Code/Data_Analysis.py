import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots folder 

os.makedirs("plots", exist_ok=True)

# Load data

df = pd.read_csv('Data/HAZOP_DATA.csv')
features = ['temperature', 'pressure', 'flow_rate', 'level', 'vibration']

# Risk distribution

plt.figure(figsize=(6,6))
df['risk'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Low', 'Medium', 'High'])
plt.title('Risk Distribution')
plt.savefig('Plots/risk_distribution.png')
plt.close()

# Feature Distribution

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, f in enumerate(features):
    sns.histplot(df[f], kde=True, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f)
axes[1,2].axis('off')
plt.tight_layout()
plt.savefig('Plots/feature_distribution.png')
plt.close()

# Correlation Heatmap

plt.figure(figsize=(10,8))
sns.heatmap(df[features + ['risk']].corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.savefig('Plots/correlation_matrix.png')
plt.close()

print("Analysis plots saved in 'Plots/'")