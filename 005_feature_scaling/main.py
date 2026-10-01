import numpy as np

from linear_regression import LinearRegression
from scaler import StandardScaler


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


# Create scaler
scaler = StandardScaler()

# Scale features
X_scaled = scaler.fit_transform(X)


print("Original X:")
print(X)

print("\nMean:")
print(scaler.mean)

print("\nStandard deviation:")
print(scaler.std)

print("\nScaled X:")
print(X_scaled)


# Create model
model = LinearRegression(
    learning_rate=0.1,
    epochs=2000
)

# Train
model.fit(X_scaled, y)


print("\nFinal weights:")
print(model.w)

print("\nFinal bias:")
print(model.b)

print("\nPredictions:")
print(model.predict(X_scaled))

print("\nActual:")
print(y)

print("\nFinal MSE:")
print(model.loss_history[-1])