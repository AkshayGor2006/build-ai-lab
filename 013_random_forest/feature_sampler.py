import numpy as np


class FeatureSampler:

    def __init__(self, max_features=None, random_state=None):
        self.max_features = max_features
        self.random_state = random_state

    def sample(self, X):

        number_of_features = X.shape[1]

        if self.max_features is None:
            max_features = int(np.sqrt(number_of_features))
        else:
            max_features = self.max_features

        rng = np.random.default_rng(self.random_state)

        feature_indices = rng.choice(
            number_of_features,
            size=max_features,
            replace=False
        )

        feature_indices.sort()

        X_sample = X[:, feature_indices]

        return X_sample, feature_indices