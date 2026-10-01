import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 2],

    [7, 7],
    [7, 8],
    [8, 7],
    [8, 8],
    [9, 9]
], dtype=float)

y = np.array([
    0,
    0,
    0,
    0,
    0,

    1,
    1,
    1,
    1,
    1
])

x1 = np.linspace(0, 10, 100)
x2 = np.linspace(0, 10, 100)

xx, yy = np.meshgrid(x1, x2)

grid_points = np.column_stack([
    xx.ravel(),
    yy.ravel()
])

print("x1 shape:", x1.shape)
print("x2 shape:", x2.shape)

print("xx shape:", xx.shape)
print("yy shape:", yy.shape)

print("grid shape:", grid_points.shape)

from knn import KNNClassifier

model = KNNClassifier(k=7)

model.fit(X, y)

grid_predictions = model.predict(grid_points)

print("Grid predictions shape:", grid_predictions.shape)

Z = grid_predictions.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    edgecolors="black"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("KNN Decision Boundary (K=7)")

plt.show()

print("Z shape:", Z.shape)