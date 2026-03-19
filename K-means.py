import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, weights=[0.9, 0.1], random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
kmeans.fit(X_scaled[y==0])

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

dists = kmeans.transform(np.c_[xx.ravel(), yy.ravel()])
Z = np.min(dists, axis=1)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Reds, levels=20)
plt.colorbar(label='Odległość od najbliższego centrum')
plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], c='blue', edgecolors='k', alpha=0.6)
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], c='red', edgecolors='k', marker='X', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', edgecolors='k', marker='*', label='Centra Klastrów')
plt.title("K-Średnie")
plt.legend()
plt.show()