import numpy as np

from knn import KNNClassifier
from train_test_split import train_test_split
from metrics import accuracy, confusion_matrix, precision, recall, f1_score


# --------------------------------------------------
# 1. Dataset
# --------------------------------------------------

X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [2, 3],
    [3, 2],
    [3, 3],
    [4, 3],

    [6, 6],
    [7, 6],
    [6, 7],
    [7, 7],
    [8, 7],
    [7, 8],
    [8, 8],
    [9, 9],
    [5, 6],
    [6, 5]
], dtype=float)


y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])


# --------------------------------------------------
# 2. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 3. Train KNN
# --------------------------------------------------

model = KNNClassifier(k=3)

model.fit(X_train, y_train)


# --------------------------------------------------
# 4. Predictions
# --------------------------------------------------

predictions = model.predict(X_test)

print("\nPredictions:", predictions)
print("Actual:     ", y_test)


# --------------------------------------------------
# 5. Metrics
# --------------------------------------------------

acc = accuracy(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
p = precision(y_test, predictions)
r = recall(y_test, predictions)
f1 = f1_score(y_test, predictions)


print("\nAccuracy:", acc)
print("Accuracy %:", acc * 100)

print("\nConfusion Matrix:")
print(cm)

print("\nPrecision:", p)
print("Recall:", r)
print("F1 Score:", f1)