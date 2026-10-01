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


# -------------------------
# 1. Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# 2. WRONG:
#    scaler sees TRAIN + TEST
# -------------------------

scaler = StandardScaler()

scaler.fit(X)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------
# 3. Train
# -------------------------

model = LinearRegression(
    learning_rate=0.1,
    epochs=2000
)

model.fit(
    X_train_scaled,
    y_train
)


# -------------------------
# 4. Test
# -------------------------

test_predictions = model.predict(
    X_test_scaled
)

test_mse = np.mean(
    (test_predictions - y_test) ** 2
)


print("WARNING: DATA LEAKAGE EXPERIMENT")

print("\nScaler mean:")
print(scaler.mean)

print("\nScaler std:")
print(scaler.std)

print("\nTest predictions:")
print(test_predictions)

print("\nActual:")
print(y_test)

print("\nTest MSE:")
print(test_mse)