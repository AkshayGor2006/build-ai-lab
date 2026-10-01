import numpy as np


class KFold:

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X):

        n_samples = len(X)

        indices = np.arange(n_samples)

        fold_sizes = np.full(
            self.n_splits,
            n_samples // self.n_splits
        )

        remainder = n_samples % self.n_splits

        fold_sizes[:remainder] += 1

        current = 0

        for fold_size in fold_sizes:

            start = current
            stop = current + fold_size

            validation_indices = indices[start:stop]

            train_indices = np.concatenate([
                indices[:start],
                indices[stop:]
            ])

            yield train_indices, validation_indices

            current = stop