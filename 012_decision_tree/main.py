import numpy as np

from decision_tree import DecisionTreeClassifier


# =========================
# Dataset
# =========================

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8]
])

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    0
])


# =========================
# Test different tree depths
# =========================

for depth in [1, 2, 3, 5, 10]:

    print("\n========================")
    print("Max Depth:", depth)
    print("========================")

    model = DecisionTreeClassifier(max_depth=depth)

    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = model.accuracy(X, y)

    print("Predictions:", predictions)
    print("Actual:     ", y)
    print("Accuracy:   ", accuracy)
    print("Accuracy %: ", accuracy * 100)