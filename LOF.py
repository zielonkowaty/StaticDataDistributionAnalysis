import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import LocalOutlierFactor

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, weights=[0.9, 0.1], random_state=42)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X[y==0])

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = lof.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
plt.colorbar(label='Wynik LOF (Niższy == Bardziej nietypowy)')
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', edgecolors='k')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', edgecolors='k', marker='X', s=100)
plt.title("LOF")
plt.show()