import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = 0.0
        self.b = 0.0

        self.loss_history = []

    def predict(self, X):

        return self.w * X + self.b

    def fit(self, X, y, verbose=False):

        for epoch in range(self.epochs):

            # 1. Prediction
            predictions = self.predict(X)

            # 2. Error
            errors = predictions - y

            # 3. Loss
            mse = np.mean(errors ** 2)

            # 4. Gradients
            gradient_w = (2 / len(X)) * np.sum(errors * X)
            gradient_b = (2 / len(X)) * np.sum(errors)

            # 5. Update parameters
            self.w -= self.learning_rate * gradient_w
            self.b -= self.learning_rate * gradient_b

            # 6. Store loss
            self.loss_history.append(mse)

            # 7. Training logs
            if verbose and epoch % 100 == 0:
                print(
                    f"Epoch {epoch}: "
                    f"MSE={mse:.4f}, "
                    f"w={self.w:.4f}, "
                    f"b={self.b:.4f}"
                )

        return self

    def evaluate(self, X, y):

        predictions = self.predict(X)

        errors = predictions - y

        mse = np.mean(errors ** 2)

        return mse

    def summary(self):

        print("\nModel Summary")
        print("--------------------")
        print("Learning rate:", self.learning_rate)
        print("Epochs:", self.epochs)
        print("Weight (w):", self.w)
        print("Bias (b):", self.b)

        if self.loss_history:
            print("Initial MSE:", self.loss_history[0])
            print("Final MSE:", self.loss_history[-1])