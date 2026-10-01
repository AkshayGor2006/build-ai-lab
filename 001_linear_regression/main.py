import numpy as np

from linear_regression import LinearRegression


# -------------------------
# Dataset
# -------------------------

X = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8],
    dtype=float
)

y = np.array(
    [35, 42, 51, 59, 68, 75, 84, 91],
    dtype=float
)


# -------------------------
# Create model
# -------------------------

model = LinearRegression(
    learning_rate=0.01,
    epochs=1000
)


# -------------------------
# Train
# -------------------------

model.fit(X, y, verbose=True)


# -------------------------
# Inspect model
# -------------------------

model.summary()


# -------------------------
# Evaluate
# -------------------------

mse = model.evaluate(X, y)

print("\nEvaluation MSE:", mse)


# -------------------------
# Predict
# -------------------------

prediction = model.predict(9)

print("Prediction for 9 hours:", prediction)

import matplotlib.pyplot as plt

plt.plot(model.loss_history)

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training Loss")

plt.show()