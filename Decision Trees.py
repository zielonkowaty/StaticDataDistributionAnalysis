import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.9, 0.1], random_state=42)

clf = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu_r)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', edgecolors='k', label='Norma')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', edgecolors='k', marker='X', s=100, label='Anomalia')
plt.title("Drzewo Decyzyjne")
plt.legend()
plt.show()