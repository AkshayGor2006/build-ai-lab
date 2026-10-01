import numpy as np

X = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8]
], dtype=float)

# Step 1: Calculate feature means
mean = np.mean(X, axis=0)

# Step 2: Center the data
X_centered = X - mean

# Step 3: Calculate covariance matrix
covariance_matrix = (
    X_centered.T @ X_centered
) / (len(X) - 1)

print("Original X:")
print(X)

print("\nMean:")
print(mean)

print("\nCentered X:")
print(X_centered)

print("\nCovariance Matrix:")
print(covariance_matrix)

eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Sort eigenvalues from largest to smallest
sorted_indices = np.argsort(eigenvalues)[::-1]

print("\nSorted indices:")
print(sorted_indices)

sorted_eigenvectors = eigenvectors[:, sorted_indices]

print("\nSorted eigenvectors:")
print(sorted_eigenvectors)

n_components = 1

principal_components = sorted_eigenvectors[:, :n_components]

print("\nSelected Principal Component:")
print(principal_components)

X_reduced = X_centered @ principal_components

print("\nReduced Data:")
print(X_reduced)

print("\nOriginal Shape:")
print(X.shape)

print("\nReduced Shape:")
print(X_reduced.shape)

total_variance = np.sum(eigenvalues)

print("\nTotal Variance:")
print(total_variance)

explained_variance_ratio = eigenvalues / total_variance

print("\nExplained Variance Ratio:")
print(explained_variance_ratio)

sorted_explained_variance_ratio = explained_variance_ratio[sorted_indices]

print("\nSorted Explained Variance Ratio:")
print(sorted_explained_variance_ratio)

selected_variance = sorted_explained_variance_ratio[:n_components]

print("\nSelected Explained Variance:")
print(selected_variance)

print("\nTotal Variance Preserved:")
print(np.sum(selected_variance))