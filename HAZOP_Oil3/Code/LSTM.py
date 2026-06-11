# LSTM For Time Series Risk Prediction

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONDCNN_OPTS'] = '0'

# Load data

df = pd.read_csv('Data/HAZOP_DATA.csv')
features = ['temperature', 'pressure', 'flow_rate', 'level', 'vibration']
X = df[features].values
y = df['risk'].values

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

# Train & test split

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

print(f"LSTM DATA SHAPE")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

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
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

loss, acc = model.evaluate(X_test, y_test)
print(f" LSTM Test Accuracy: {acc:.4f}")
model.save('lstm.model.keras')