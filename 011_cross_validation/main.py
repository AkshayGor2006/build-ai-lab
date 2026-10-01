import numpy as np

from kfold import KFold
from knn import KNNClassifier
from metrics import accuracy


X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [3, 2],
    [7, 7],
    [7, 8],
    [8, 7],
    [8, 8],
    [9, 9]
], dtype=float)

y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])

k_values = [1, 3, 5, 7]

kf = KFold(n_splits=5)

for k in k_values:

    scores = []

    print("\n======================")
    print(f"K = {k}")
    print("======================")

    for fold, (train_indices, validation_indices) in enumerate(
        kf.split(X),
        start=1
    ):

        X_train = X[train_indices]
        y_train = y[train_indices]

        X_validation = X[validation_indices]
        y_validation = y[validation_indices]

        model = KNNClassifier(k=k)

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        score = accuracy(
            y_validation,
            predictions
        )

        scores.append(score)

        print(
            f"Fold {fold}: "
            f"Accuracy = {score:.2f}"
        )

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    print(f"Mean Accuracy: {mean_accuracy:.2f}")
    print(f"Std Accuracy: {std_accuracy:.2f}")