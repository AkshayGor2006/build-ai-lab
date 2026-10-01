import numpy as np

from kmeans import KMeans


X = np.array([
    [1, 1],
    [1.5, 2],
    [2, 1.5],
    [8, 8],
    [8.5, 9],
    [9, 8.5]
], dtype=float)


model = KMeans(
    k=3,
    max_iterations=100,
    random_state=42
)

model.fit(X)


print("\nFinal Centroids:")
print(model.centroids)

print("\nCluster Labels:")
print(model.labels)

print("\nInertia:")
print(model.calculate_inertia(X))


X_new = np.array([
    [1.2, 1.5],
    [8.2, 8.5]
], dtype=float)


predictions = model.predict(X_new)

print("\nNew Predictions:")
print(predictions)