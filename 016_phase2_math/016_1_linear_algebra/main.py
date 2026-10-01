import numpy as np


# -------------------------
# Vector
# -------------------------

v = np.array([3, 4], dtype=float)

print("Vector:")
print(v)


# -------------------------
# Norm
# -------------------------

norm = np.sqrt(np.sum(v ** 2))

print("\nNorm:")
print(norm)


# -------------------------
# Normalization
# -------------------------

normalized_v = v / norm

print("\nNormalized Vector:")
print(normalized_v)

print("\nNormalized Norm:")
print(np.sqrt(np.sum(normalized_v ** 2)))


# -------------------------
# Projection
# -------------------------

a = np.array([3, 4], dtype=float)
b = np.array([1, 0], dtype=float)

projection = (np.dot(a, b) / np.dot(b, b)) * b

print("\nVector A:")
print(a)

print("\nVector B:")
print(b)

print("\nProjection of A onto B:")
print(projection)