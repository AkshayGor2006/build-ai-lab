import numpy as np


class KMeans:

    def __init__(self, k=2, max_iterations=100, random_state=None):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state

        self.centroids = None
        self.labels = None
        self.inertia = None
        

    def fit(self, X):

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Step 1: Initialize centroids
        indices = np.random.choice(
            len(X),
            size=self.k,
            replace=False
        )

        self.centroids = X[indices].copy()

        for iteration in range(self.max_iterations):

            # Step 2: Calculate distances
            distances = self.calculate_distances(X)

            # Step 3: Assign each point to nearest centroid
            new_labels = np.argmin(distances, axis=1)

            # Step 4: Recalculate centroids
            new_centroids = np.zeros_like(self.centroids)

            for cluster in range(self.k):

                cluster_points = X[new_labels == cluster]

                if len(cluster_points) > 0:
                    new_centroids[cluster] = np.mean(
                        cluster_points,
                        axis=0
                    )
                else:
                    new_centroids[cluster] = self.centroids[cluster]

            # Step 5: Check convergence
            if np.allclose(self.centroids, new_centroids):
                self.centroids = new_centroids
                self.labels = new_labels

                print(f"Converged at iteration {iteration}")

                break

            self.centroids = new_centroids
            self.labels = new_labels

            self.inertia = 0.0

            for i in range(len(X)):
                distance = np.linalg.norm(
                X[i] - self.centroids[self.labels[i]]
                )

            self.inertia += distance ** 2

        return self

    def calculate_distances(self, X):

        distances = np.zeros((len(X), self.k))

        for i, point in enumerate(X):

            for j, centroid in enumerate(self.centroids):

                distances[i, j] = np.sqrt(
                    np.sum((point - centroid) ** 2)
                )

        return distances

    def predict(self, X):

        distances = self.calculate_distances(X)

        return np.argmin(distances, axis=1)

    def calculate_inertia(self, X):

        distances = self.calculate_distances(X)

        assigned_distances = distances[
            np.arange(len(X)),
            self.labels
        ]

        inertia = np.sum(assigned_distances ** 2)

        return inertia