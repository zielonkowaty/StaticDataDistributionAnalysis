import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, weights=[0.9, 0.1], random_state=42)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_normal = X_scaled[y == 0]

input_dim = 2
input_layer = Input(shape=(input_dim,))
encoder = Dense(1, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_normal, X_normal, epochs=100, batch_size=16, verbose=0)

x_min, x_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
y_min, y_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]

reconstructions = autoencoder.predict(grid_points)
mse = np.mean(np.power(grid_points - reconstructions, 2), axis=1)
Z = mse.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Reds, levels=20)
plt.colorbar(label='Błąd Rekonstrukcji (MSE)')
plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], c='blue', edgecolors='k', alpha=0.6)
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], c='red', edgecolors='k', marker='X', s=100)
plt.title("Autoenkoder")
plt.show()