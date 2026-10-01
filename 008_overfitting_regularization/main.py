import numpy as np

from polynomial_regression import PolynomialRegression


# --------------------------------
# Dataset
# --------------------------------

np.random.seed(42)

X = np.linspace(-3, 3, 30)

y = (
    2 * X**3
    - 3 * X**2
    + 2 * X
    + 5
    + np.random.normal(0, 3, len(X))
)


# --------------------------------
# Train/Test Split
# --------------------------------

indices = np.random.permutation(len(X))

split_index = int(len(X) * 0.8)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------
# Model
# --------------------------------

model = PolynomialRegression(
    degree=15,
    learning_rate=0.000001,
    epochs=10000,
    reg_lambda=0.0
)


# --------------------------------
# Train
# --------------------------------

model.fit(X_train, y_train)


# --------------------------------
# Evaluate
# --------------------------------

train_mse = model.evaluate(
    X_train,
    y_train
)

test_mse = model.evaluate(
    X_test,
    y_test
)


print("\nTraining MSE:")
print(train_mse)

print("\nTest MSE:")
print(test_mse)


print("\nWeights:")
print(model.w)

print("\nBias:")
print(model.b)

print("Weights:")
print(model.w)