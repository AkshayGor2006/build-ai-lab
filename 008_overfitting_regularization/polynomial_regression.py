import numpy as np


class PolynomialRegression:

    def __init__(
        self,
        degree=1,
        learning_rate=0.00001,
        epochs=5000,
        reg_lambda=0.0
    ):
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda

        self.w = None
        self.b = 0.0
        self.loss_history = []


    def create_features(self, X):

        X = np.asarray(X).reshape(-1, 1)

        features = []

        for power in range(1, self.degree + 1):
            features.append(X[:, 0] ** power)

        return np.column_stack(features)


    def predict(self, X):

        X_poly = self.create_features(X)

        return X_poly @ self.w + self.b


    def fit(self, X, y):

        X_poly = self.create_features(X)

        number_of_features = X_poly.shape[1]

        self.w = np.zeros(number_of_features)

        for epoch in range(self.epochs):

            predictions = self.predict(X)

            errors = predictions - y

            mse = np.mean(errors ** 2)

            regularization = (
                self.reg_lambda * np.sum(self.w ** 2)
            )

            loss = mse + regularization

            gradient_w = (
                (2 / len(X_poly)) * (X_poly.T @ errors)
                + 2 * self.reg_lambda * self.w
            )

            gradient_b = (
                (2 / len(X_poly)) * np.sum(errors)
            )

            self.w -= self.learning_rate * gradient_w

            self.b -= self.learning_rate * gradient_b

            self.loss_history.append(loss)

            if epoch % 500 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"Loss={loss:.4f}"
                )

        return self


    def evaluate(self, X, y):

        predictions = self.predict(X)

        errors = predictions - y

        mse = np.mean(errors ** 2)

        return mse