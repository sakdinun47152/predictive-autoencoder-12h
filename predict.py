import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models

def set_seeds(seed=650610240):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(650612126)

try:
    df = pd.read_csv('load_consumption.csv')
    data = df['Load (kWh)'].values.reshape(-1, 1)
except FileNotFoundError: 
    print("Error: 'load_consumption.csv' not found. Please check your file path.")
    exit()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

LOOK_BACK = 5
FORECAST = 12

model = models.load_model('models/650612126.keras')

last_sequence = scaled_data[-LOOK_BACK:].reshape(1, -1)
predicted_scaled = model.predict(last_sequence)
predicted_load = scaler.inverse_transform(predicted_scaled)

hours_ahead = np.arange(1, FORECAST + 1)
plt.plot(hours_ahead, predicted_load[0], marker='o', color='green', label='Forecast')
plt.title(f'Load Forecast for next {FORECAST} Hours')
plt.xlabel('Hours Ahead')
plt.ylabel('Load (kWh)')
plt.xticks(hours_ahead)
plt.legend()
plt.tight_layout()
plt.show()

hours = [f"{i:02d}:00" for i in range(1, 13)]
for h, val in zip(hours, predicted_load[0]):
    print(f"{h} -> {val:.2f} kWh")