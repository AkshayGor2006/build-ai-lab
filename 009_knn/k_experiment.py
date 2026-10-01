import numpy as np

from knn import KNNClassifier
from train_test_split import train_test_split
from metrics import accuracy, precision, recall, f1_score


# --------------------------------------------------
# Dataset
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
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)


# --------------------------------------------------
# Test different K values
# --------------------------------------------------

k_values = [1, 3, 5, 7]

for k in k_values:

    model = KNNClassifier(k=k)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy(y_test, predictions)
    p = precision(y_test, predictions)
    r = recall(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n--------------------------")
    print("K =", k)
    print("--------------------------")

    print("Predictions:", predictions)
    print("Actual:     ", y_test)

    print("Accuracy:", acc)
    print("Precision:", p)
    print("Recall:", r)
    print("F1 Score:", f1)