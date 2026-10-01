import numpy as np

from train_test_split import train_test_split
from scaler import StandardScaler
from linear_regression import LinearRegression


X = np.array([
    [1000, 2, 20],
    [1200, 2, 15],
    [1400, 3, 12],
    [1500, 3, 10],
    [1600, 3, 8],
    [1800, 4, 7],
    [2000, 4, 5],
    [2200, 4, 4],
    [2400, 5, 3],
    [2600, 5, 2]
], dtype=float)


y = np.array([
    500,
    580,
    650,
    720,
    760,
    850,
    950,
    1020,
    1120,
    1200
], dtype=float)


# 1. Split the data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# 2. Create scaler

scaler = StandardScaler()


# 3. FIT only on training data

X_train_scaled = scaler.fit_transform(X_train)


# 4. Transform test data using
#    training mean and std

X_test_scaled = scaler.transform(X_test)


print("\nTraining mean:")
print(scaler.mean)

print("\nTraining std:")
print(scaler.std)


# 5. Create model

model = LinearRegression(
    learning_rate=0.1,
    epochs=2000
)


# 6. Train ONLY on training data

model.fit(X_train_scaled, y_train)


# 7. Predictions on training data

train_predictions = model.predict(X_train_scaled)


# 8. Predictions on unseen test data

test_predictions = model.predict(X_test_scaled)


# 9. Calculate MSE

train_mse = np.mean(
    (train_predictions - y_train) ** 2
)

test_mse = np.mean(
    (test_predictions - y_test) ** 2
)


print("\nTraining MSE:")
print(train_mse)

print("\nTest MSE:")
print(test_mse)

print("\nTest predictions:")
print(test_predictions)

print("\nActual test values:")
print(y_test)