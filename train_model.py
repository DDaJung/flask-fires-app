import numpy as np
import tensorflow as tf

X = np.load("model/X_train.npy")
y = np.load("model/y_train.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, validation_split=0.2)

model.save("model/wildfire_predictor.h5")
