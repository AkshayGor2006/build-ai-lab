import numpy as np

X = np.array([
    [2,1],
    [3,1],
    [1,3],
    [1,4]
])

y = np.array([1,1,0,0])

print("X:")
print(X)
print("\ny:")
print(y)

weights = np.array([0.0, 0.0])
bias = 0.0

print("\nweights:")
print(weights)
print("\nbias:")
print(bias)

scores = X @ weights + bias

print("\nscores:")
print(scores)

predictions = (scores >= 0).astype(int)

print("\npredictions:")
print(predictions)

learning_rate = 1.0

for i in range(len(X)):
    error = y[i] - predictions[i]

    weights = weights + learning_rate * error * X[i]
    bias = bias + learning_rate * error

print("\nupdated weights:")
print(weights)

print("\nupdated bias:")
print(bias)

new_scores = X @ weights + bias
new_predictions = (new_scores >= 0).astype(int)

print("\nnew scores:")
print(new_scores)

print("\nnew predictions:")
print(new_predictions)

print("\nactual:")
print(y)

weights = np.array([0.0, 0.0])
bias = 0.0

learning_rate = 1.0
epochs = 10

for epoch in range(epochs):

    for i in range(len(X)):

        score = X[i] @ weights + bias

        prediction = 1 if score >= 0 else 0

        error = y[i] - prediction

        weights = weights + learning_rate * error * X[i]
        bias = bias + learning_rate * error

    print(
        f"Epoch {epoch + 1}: "
        f"weights={weights}, bias={bias}"
    )

# Final evaluation
final_scores = X @ weights + bias
final_predictions = (final_scores >= 0).astype(int)

correct = np.sum(final_predictions == y)
accuracy = correct / len(y)

print("\nFinal scores:")
print(final_scores)

print("\nFinal predictions:")
print(final_predictions)

print("\nActual:")
print(y)

print("\nAccuracy:")
print(accuracy)

(final_scores >= 0).astype(int)

# Test on unseen data

X_test = np.array([
    [4, 1],
    [1, 5],
    [3, 2],
    [2, 3]
])

y_test = np.array([1, 0, 1, 0])

test_scores = X_test @ weights + bias
test_predictions = (test_scores >= 0).astype(int)

test_accuracy = np.mean(test_predictions == y_test)

print("\nTest scores:")
print(test_scores)

print("\nTest predictions:")
print(test_predictions)

print("\nTest actual:")
print(y_test)

print("\nTest accuracy:")
print(test_accuracy)

print("\n--- XOR EXPERIMENT ---")

X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([0, 1, 1, 0])

xor_weights = np.array([0.0, 0.0])
xor_bias = 0.0

learning_rate = 1.0

for epoch in range(10):

    for i in range(len(X_xor)):

        score = X_xor[i] @ xor_weights + xor_bias

        prediction = 1 if score >= 0 else 0

        error = y_xor[i] - prediction

        xor_weights = xor_weights + learning_rate * error * X_xor[i]

        xor_bias = xor_bias + learning_rate * error

    print(
        f"Epoch {epoch + 1}: "
        f"weights={xor_weights}, bias={xor_bias}"
    )

xor_scores = X_xor @ xor_weights + xor_bias

xor_predictions = (xor_scores >= 0).astype(int)

print("\nXOR predictions:")
print(xor_predictions)

print("\nXOR actual:")
print(y_xor)