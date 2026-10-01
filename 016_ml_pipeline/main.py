import numpy as np

from train_test_split import train_test_split
from scaler import StandardScaler
from knn import KNN


X = np.array([
    [6, 7],
    [1, 2],
    [8, 9],
    [3, 4],
    [10, 11],
    [5, 6],
    [4, 5],
    [7, 8],
    [9, 10],
    [2, 3]
], dtype=float)

y = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])


# 1. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 2. Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. Model
model = KNN(k=3)


# 4. Training
model.fit(X_train_scaled, y_train)


# 5. Prediction
predictions = model.predict(X_test_scaled)


# 6. Evaluation
accuracy = np.mean(predictions == y_test)


print("\nActual:")
print(y_test)

print("\nPredictions:")
print(predictions)

print("\nAccuracy:")
print(accuracy)