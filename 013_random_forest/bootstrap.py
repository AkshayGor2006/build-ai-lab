import numpy as np

class BootstrapSampler:

    def __init__(self, random_state = None):
        self.random_state = random_state

    def sample(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples = len(X)

        indices = np.random.choice(
            n_samples,
            size = n_samples,
            replace=True
        )

        X_sample = X[indices]
        y_sample = y[indices]

        return X_sample, y_sample, indices