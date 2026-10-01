import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.1, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def predict(self, X):
        return X @ self.w + self.b

    def fit(self, X, y):

        number_of_features = X.shape[1]

        self.w = np.zeros(number_of_features)

        for epoch in range(self.epochs):

            predictions = self.predict(X)

            errors = predictions - y

            mse = np.mean(errors ** 2)

            gradient_w = (2 / len(X)) * (X.T @ errors)

            gradient_b = (2 / len(X)) * np.sum(errors)

            self.w -= self.learning_rate * gradient_w

            self.b -= self.learning_rate * gradient_b

            self.loss_history.append(mse)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: MSE={mse:.4f}"
                )

        return self