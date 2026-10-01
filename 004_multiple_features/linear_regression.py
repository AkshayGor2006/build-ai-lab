import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.000001, epochs=1000):

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

            # Prediction
            predictions = self.predict(X)

            # Error
            errors = predictions - y

            # Loss
            mse = np.mean(errors ** 2)

            # Gradients
            gradient_w = (2 / len(X)) * (X.T @ errors)
            gradient_b = (2 / len(X)) * np.sum(errors)

            # Update
            self.w -= self.learning_rate * gradient_w
            self.b -= self.learning_rate * gradient_b

            # Store loss
            self.loss_history.append(mse)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"MSE={mse:.4f}"
                )

        return self