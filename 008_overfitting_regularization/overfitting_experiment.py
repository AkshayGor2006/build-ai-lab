import numpy as np

from train_test_split import train_test_split
from polynomial_regression import PolynomialRegression


# --------------------------------
# 1. Create dataset
# --------------------------------

np.random.seed(42)

X = np.linspace(-1, 1, 30)

noise = np.random.normal(0, 0.25, size=30)

y = 2 * X + 1 + noise


# --------------------------------
# 2. Train / test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------
# 3. Test different model degrees
# --------------------------------

reg_lambdas = [0, 0.001, 0.01, 0.1, 1.0]

for reg_lambda in reg_lambdas:

    model = PolynomialRegression(
        degree=15,
        learning_rate=0.01,
        epochs=10000,
        reg_lambda=reg_lambda
    )

    model.fit(X_train, y_train)

    train_mse = model.evaluate(X_train, y_train)
    test_mse = model.evaluate(X_test, y_test)

    print("\n--------------------")
    print("Lambda:", reg_lambda)
    print("Training MSE:", train_mse)
    print("Test MSE:", test_mse)