import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = None
        self.b = 0.0

        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def fit(self, X, y):

        number_of_features = X.shape[1]

        self.w = np.zeros(number_of_features)
        self.b = 0.0

        for epoch in range(self.epochs):

            # Forward pass
            probabilities = self.predict_proba(X)

            # Avoid log(0)
            epsilon = 1e-15
            probabilities = np.clip(
                probabilities,
                epsilon,
                1 - epsilon
            )

            # Binary Cross-Entropy
            loss = -np.mean(
                y * np.log(probabilities)
                + (1 - y) * np.log(1 - probabilities)
            )

            # Gradients
            errors = probabilities - y

            gradient_w = (1 / len(X)) * (X.T @ errors)
            gradient_b = (1 / len(X)) * np.sum(errors)

            # Update parameters
            self.w -= self.learning_rate * gradient_w
            self.b -= self.learning_rate * gradient_b

            self.loss_history.append(loss)

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"Loss={loss:.4f}"
                )

        return self