import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

t = np.linspace(0, 50, 500)
y = np.sin(t)

y[200:250] = 0.5
y[400:420] = -1.5

X = y.reshape(-1, 1)

model = IsolationForest(contamination=0.15, random_state=42)
model.fit(X)
anomalies = model.predict(X)

plt.figure(figsize=(10, 6))
plt.plot(t, y, label='Dane', color='blue')
plt.scatter(t[anomalies == -1], y[anomalies == -1], color='red', label='Anomalie')
plt.legend()
plt.title("Las Izolujący")
plt.show()