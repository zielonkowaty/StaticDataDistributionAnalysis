import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

t = np.linspace(0, 50, 600)
signal = np.sin(t)

signal[300:400] = 0.0 + np.random.normal(0, 0.05, 100)

window_size = 10
X_windows = []
for i in range(len(signal) - window_size):
    X_windows.append(signal[i:i+window_size])
X_windows = np.array(X_windows)

model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
model.fit(X_windows)
y_pred = model.predict(X_windows)

anomaly_indices = np.where(y_pred == -1)[0]
anomaly_indices += window_size // 2

plt.figure(figsize=(12, 6))
plt.plot(t, signal, label='Sygnał', color='green')
plt.scatter(t[anomaly_indices], signal[anomaly_indices], color='red', s=10, label='Anomalia Zbiorowa')
plt.title("Detekcja Anomalii Zbiorowych")
plt.legend()
plt.show()