import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, layers, optimizers, callbacks

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

X, y = [], []
for i in range(len(scaled_data) - LOOK_BACK - FORECAST + 1):
    X.append(scaled_data[i : i + LOOK_BACK].flatten())
    y.append(scaled_data[i + LOOK_BACK : i + LOOK_BACK + FORECAST].flatten())

X = np.array(X)
y = np.array(y)

total_samples = len(X)
train_end = int(total_samples * 0.7)
val_end = int(total_samples * 0.85)

X_train, y_train = X[:train_end], y[:train_end]

X_val, y_val = X[train_end:val_end], y[train_end:val_end]

X_test, y_test = X[val_end:], y[val_end:]

model = models.Sequential([
    layers.Input(shape=(LOOK_BACK,)),
    
    layers.Dense(8, activation='relu', name='Encoding_Feature_Extractor'),
    layers.Dense(3, activation='relu', name='Latent_Space'),
    
    layers.Dropout(0.3), 
    
    layers.Dense(8, activation='relu', name='Decoding_Expansion_Layer'),
    layers.Dense(FORECAST, activation='linear', name='Predictor')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
model.summary()

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

print("\nTraining Predictive Autoencoder...")
history = model.fit(
    X, y, 
    epochs=1000, 
    batch_size=64, 
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stop]
)

test_loss = model.evaluate(X_test, y_test)
print(f"\nTest Loss (MSE): {test_loss:.6f}")

test_prediction = model.predict(X_test[0:1])
test_prediction_original = scaler.inverse_transform(test_prediction)
actual_original = scaler.inverse_transform(y_test[0:1])

print("\nComparison on Test Sample:")
print(f"Actual: {actual_original[0][:5]} ...")
print(f"Predicted: {test_prediction_original[0][:5]} ...")

plt.figure(figsize=(6, 4))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
plt.title('Model Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

user_input = input('\nSave? (Y/N): ').lower()

if user_input == "yes" or user_input == "y":
    name = input('Name: ')
    print("Continuing...")
    model.save(f'models/{name}.keras')
    print(f"{name}.keras have been saved.")
elif user_input == "no" or user_input == "n":
    print("Exiting...")
else:
    print("Invalid input. Please enter Y or N.")