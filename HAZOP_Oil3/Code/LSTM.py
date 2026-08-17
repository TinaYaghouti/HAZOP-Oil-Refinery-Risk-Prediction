# LSTM For Time Series Risk Prediction

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONDCNN_OPTS'] = '0'

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(base_dir, 'Data', 'HAZOP_DATA.csv'))

# Load data

# df = pd.read_csv('Data/HAZOP_DATA.csv')
features = ['temperature', 'pressure', 'flow_rate', 'level', 'vibration']
X = df[features].values
y = df['risk'].values

# Check class balance first 

print("Class distribution:")
print(pd.Series(y).value_counts(normalize=True))

# Normalize

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequances for LSTM

def CreateSequances(data, labels, seq_length=10):
    Xs, ys = [], []
    for i in range(len(data) - seq_length):
        Xs.append(data[i:i+seq_length])
        ys.append(labels[i+seq_length])
    return np.array(Xs), np.array(ys)

SEQ = 10
X_seq, y_seq = CreateSequances(X_scaled, y, SEQ)

# Time-based split ( NOT random ), sequences overlap, so random split leaks data

split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

val_split_idx = int(len(X_train) * 0.8)
X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]

from imblearn.over_sampling import SMOTE

n_samples, n_steps, n_features = X_train.shape
X_train_2d = X_train.reshape(n_samples, n_steps * n_features)

y_train = y_train.ravel()

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_2d, y_train)

X_train_bal = X_train_bal.reshape(-1, n_steps, n_features)

# print("After SMOTE:")
# print(pd.Series(y_balanced).value_counts())

print(f"\nLSTM DATA SHAPE")
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# Compute class weights to fix imbalance

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}")

# Build LSTM model

model = Sequential([
    Input(shape=(SEQ, len(features))),
    LSTM(64, return_sequences=True), Dropout(0.2),
    LSTM(32, return_sequences=False), Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nLSTM Test Accuracy: {acc:.4f}")
model.save('lstm.model.keras')

# Plot training history

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch'); plt.ylabel('accuracy'); plt.legend()

plt.tight_layout()
plt.savefig('HAZOP_OIL3/plots/lstm_training.png')
plt.show()