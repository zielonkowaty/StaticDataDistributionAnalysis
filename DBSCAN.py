import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
X_final = np.vstack([X, outliers])

db = DBSCAN(eps=0.8, min_samples=5)
labels = db.fit_predict(X_final)

plt.figure(figsize=(10, 6))
plt.scatter(X_final[:, 0], X_final[:, 1], c=labels, cmap='Paired', marker='o')
plt.title("DBSCAN (Punkty -1 == anomalie)")
plt.xlabel("Cecha 1")
plt.ylabel("Cecha 2")
plt.show()