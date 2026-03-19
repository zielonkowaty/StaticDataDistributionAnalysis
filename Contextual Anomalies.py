import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(40))

y[80:90] = -1.5

regr = RandomForestRegressor(n_estimators=100, random_state=42)
regr.fit(X, y)
y_pred = regr.predict(X)

residuals = np.abs(y - y_pred)
threshold = np.percentile(residuals, 95)
anomalies_idx = np.where(residuals > threshold)[0]

plt.figure(figsize=(10, 6))
plt.plot(X, y, color='gray', label='Dane rzeczywiste', alpha=0.5)
plt.plot(X, y_pred, color='blue', linewidth=2, label='Kontekst')
plt.scatter(X[anomalies_idx], y[anomalies_idx], color='red', s=50, label='Anomalie Kontekstowe')
plt.fill_between(X.ravel(), y_pred - threshold, y_pred + threshold, color='blue', alpha=0.1)
plt.title("Detekcja Anomalii Kontekstowych")
plt.legend()
plt.show()