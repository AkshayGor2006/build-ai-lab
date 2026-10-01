import numpy as np

from naive_bayes import BernoulliNaiveBayes


X = np.array([
    [1, 1],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0],
    [0, 0],
    [0, 1],
    [0, 0]
])


y = np.array([
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0
])


# -------------------------
# Train / Test Split
# -------------------------

X_train = X[:6]
y_train = y[:6]

X_test = X[6:]
y_test = y[6:]


print("Training data:")
print(X_train)

print("\nTraining labels:")
print(y_train)

print("\nTest data:")
print(X_test)

print("\nTest labels:")
print(y_test)


# -------------------------
# Train model
# -------------------------

model = BernoulliNaiveBayes()

model.fit(X_train, y_train)


# -------------------------
# Test model
# -------------------------

predictions = model.predict(X_test)


print("\nPredictions:")
print(predictions)

print("\nActual:")
print(y_test)


# -------------------------
# Evaluate
# -------------------------

accuracy = model.accuracy(X_test, y_test)

print("\nTest Accuracy:")
print(accuracy)

print("\nTest Accuracy %:")
print(accuracy * 100)

# -------------------------
# Final Naive Bayes Test
# -------------------------

X_new = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
], dtype=float)

predictions = model.predict(X_new)

print("\nFinal New Samples:")
print(X_new)

print("\nFinal Predictions:")
print(predictions)