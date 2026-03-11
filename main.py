import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, layers, optimizers

try:
    df = pd.read_csv('load_consumption.csv')
    data = df['Load (kWh)'].values.reshape(-1, 1)
except FileNotFoundError: 
    print("Error: File Not Found.")
    exit()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

X_train, y_train = [], []
for i in range(len(scaled_data) - 5 - 12 + 1):
    X_train.append(scaled_data[i : i + 5].flatten())
    y_train.append(scaled_data[i + 5 : i + 5 + 12].flatten())

X_train = np.array(X_train)
y_train = np.array(y_train)

model = models.Sequential([
    layers.Input(shape=(5,)),
    layers.Dense(3, activation='relu', name='Encoder'),
    layers.Dense(12, activation='linear', name='Predictor')
])


model.compile(optimizer=optimizers.Adam(learning_rate=.001), loss='mse')

print("\nTraning Model...")
history = model.fit(X_train, y_train, epochs=200,validation_split=0.25, verbose=2)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='dashed', linewidth=2)
plt.title('Overfitting Check')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.show()

last_5_hours = np.array([scaled_data[-5:].flatten()])
predicted_scaled = model.predict(last_5_hours)

predicted_load = scaler.inverse_transform(predicted_scaled)

hours = [f"{i:02d}:00" for i in range(1, 13)]
for h, val in zip(hours, predicted_load[0]):
    print(f"{h} -> {val:.2f} kWh")