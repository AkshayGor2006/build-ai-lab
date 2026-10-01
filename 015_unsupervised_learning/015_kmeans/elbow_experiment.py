import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans


X = np.array([
    [1, 1],
    [1.5, 2],
    [2, 1.5],
    [8, 8],
    [8.5, 9],
    [9, 8.5]
], dtype=float)


k_values = range(1, 6)

inertias = []


for k in k_values:

    model = KMeans(
        k=k,
        max_iterations=100,
        random_state=42
    )

    model.fit(X)

    inertias.append(model.inertia)

    print(f"K = {k}, Inertia = {model.inertia}")


plt.plot(k_values, inertias, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()