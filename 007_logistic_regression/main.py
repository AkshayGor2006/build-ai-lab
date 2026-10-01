import numpy as np

from logistic_regression import LogisticRegression
from metrics import (
    confusion_matrix,
    accuracy,
    precision,
    recall,
    f1_score
)


# --------------------------------
# 1. Dataset
# --------------------------------

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
    [11],
    [12]
], dtype=float)

y = np.array([
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1
])


# --------------------------------
# 2. Train/Test Split
# --------------------------------

np.random.seed(42)

indices = np.random.permutation(len(X))

split_index = int(len(X) * 0.8)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]


print("X_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)


# --------------------------------
# 3. Feature Scaling
# --------------------------------

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std


print("\nTraining mean:")
print(mean)

print("\nTraining std:")
print(std)


# --------------------------------
# 4. Train Logistic Regression
# --------------------------------

model = LogisticRegression(
    learning_rate=0.1,
    epochs=2000
)

model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------
# 5. Probability Predictions
# --------------------------------

probabilities = model.predict_proba(
    X_test_scaled
)


print("\nTest probabilities:")
print(probabilities)


# --------------------------------
# 6. Classification Threshold
# --------------------------------

threshold = 0.7

predictions = (
    probabilities >= threshold
).astype(int)


print("\nPredicted classes:")
print(predictions)

print("\nActual classes:")
print(y_test)


# --------------------------------
# 7. Classification Metrics
# --------------------------------

tp, tn, fp, fn = confusion_matrix(
    y_test,
    predictions
)


print("\nConfusion Matrix Values:")
print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)


print("\nAccuracy:")
print(accuracy(y_test, predictions))


print("\nPrecision:")
print(precision(y_test, predictions))


print("\nRecall:")
print(recall(y_test, predictions))


print("\nF1 Score:")
print(f1_score(y_test, predictions))