import numpy as np


class KNN:

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        return self

    def predict(self, X):
        predictions = []

        for x in X:
            distances = np.sqrt(
                np.sum((self.X_train - x) ** 2, axis=1)
            )

            nearest_indices = np.argsort(distances)[:self.k]

            nearest_labels = self.y_train[nearest_indices]

            values, counts = np.unique(
                nearest_labels,
                return_counts=True
            )

            prediction = values[np.argmax(counts)]

            predictions.append(prediction)

        return np.array(predictions)

    def evaluate(self, X, y):
        predictions = self.predict(X)

        accuracy = np.mean(predictions == y)

        return accuracy