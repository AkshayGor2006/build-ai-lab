import numpy as np

from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier


# -------------------------
# Dataset
# -------------------------

X = np.array([
    [1, 10, 100, 5],
    [2, 20, 200, 6],
    [3, 30, 300, 7],
    [4, 40, 400, 8],
    [5, 50, 500, 9],
    [6, 60, 600, 10],
    [7, 70, 700, 11],
    [8, 80, 800, 12],
    [9, 90, 900, 13],
    [10, 100, 1000, 14]
])

y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])


# -------------------------
# Decision Tree
# -------------------------

tree = DecisionTreeClassifier(
    max_depth=3
)

tree.fit(X, y)

tree_predictions = tree.predict(X)

tree_accuracy = tree.accuracy(X, y)


print("\n====================")
print("DECISION TREE")
print("====================")

print("Predictions:", tree_predictions)
print("Actual:     ", y)
print("Accuracy:   ", tree_accuracy)
print("Accuracy %: ", tree_accuracy * 100)


# -------------------------
# Random Forest
# -------------------------

forest = RandomForestClassifier(
    n_trees=5,
    max_depth=3,
    random_state=42
)

forest.fit(X, y)

forest_predictions = forest.predict(X)

forest_accuracy = forest.accuracy(X, y)


print("\n====================")
print("RANDOM FOREST")
print("====================")

print("Predictions:", forest_predictions)
print("Actual:     ", y)
print("Accuracy:   ", forest_accuracy)
print("Accuracy %: ", forest_accuracy * 100)

X_test = np.array([
    [1.5, 15, 150, 5.5],
    [5.5, 55, 550, 9.5],
    [8.5, 85, 850, 12.5]
])

y_test = np.array([0, 1, 1])

test_predictions = forest.predict(X_test)

test_accuracy = forest.accuracy(X_test, y_test)

print("\n====================")
print("RANDOM FOREST TEST")
print("====================")

print("Predictions:", test_predictions)
print("Actual:     ", y_test)
print("Accuracy:   ", test_accuracy)
print("Accuracy %: ", test_accuracy * 100)