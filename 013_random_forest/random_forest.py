import numpy as np

from bootstrap import BootstrapSampler
from feature_sampler import FeatureSampler
from decision_tree import DecisionTreeClassifier


class RandomForestClassifier:

    def __init__(
        self,
        n_trees=5,
        max_depth=3,
        random_state=42
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state

        self.trees = []

    def fit(self, X, y):

        self.trees = []

        for tree_number in range(self.n_trees):

            # 1. Bootstrap sampling
            bootstrap_sampler = BootstrapSampler(
                random_state=self.random_state + tree_number
            )

            X_sample, y_sample, _ = bootstrap_sampler.sample(X, y)

            # 2. Feature sampling
            feature_sampler = FeatureSampler(
                random_state=self.random_state + tree_number
            )

            X_features, feature_indices = feature_sampler.sample(X_sample)

            # 3. Train a decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth
            )

            tree.fit(X_features, y_sample)

            # 4. Store the tree and its selected features
            self.trees.append(
                (tree, feature_indices)
            )

            print(
                f"Tree {tree_number + 1}: "
                f"features={feature_indices}"
            )

        return self

    def predict(self, X):

        predictions = []

        for x in X:

            prediction = self.predict_one(x)

            predictions.append(prediction)

        return np.array(predictions)

    def accuracy(self, X, y):

        predictions = self.predict(X)

        correct = np.sum(predictions == y)

        return correct / len(y)

    def majority_vote(self, predictions):

        values, counts = np.unique(
            predictions,
            return_counts=True
        )

        majority_index = np.argmax(counts)

        return values[majority_index]

    def predict_one(self, x):

        predictions = []

        for tree, feature_indices in self.trees:

            x_selected = x[feature_indices]

            prediction = tree.predict(
                np.array([x_selected])
            )[0]

            predictions.append(prediction)

        return self.majority_vote(predictions)