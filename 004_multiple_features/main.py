import numpy as np

from linear_regression import LinearRegression


X = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1200, 2, 15],
    [1800, 3, 7]
], dtype=float)


y = np.array([
    800,
    1100,
    650,
    950
], dtype=float)


model = LinearRegression(
    learning_rate=0.000001,
    epochs=1000
)


model.fit(X, y)


print("\nFinal weights:")
print(model.w)

print("\nFinal bias:")
print(model.b)

print("\nPredictions:")
print(model.predict(X))

print("\nActual:")
print(y)

print("\nFinal MSE:")
print(model.loss_history[-1])