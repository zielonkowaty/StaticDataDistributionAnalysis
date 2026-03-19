import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, weights=[0.9, 0.1], random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svm = SVC(kernel='rbf', C=1, gamma='auto', class_weight='balanced', random_state=42)
svm.fit(X_scaled, y)

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu_r)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=200, linewidth=1, facecolors='none', edgecolors='k')
plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], c='blue', edgecolors='k')
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], c='red', edgecolors='k', marker='X', s=100)
plt.title("SVM (Zakreślone == Wektory Nośne)")
plt.show()