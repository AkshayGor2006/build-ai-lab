import numpy as np


class BernoulliNaiveBayes:

    def __init__(self):
        self.classes = None
        self.priors = {}
        self.feature_probabilities = {}

    def fit(self, X, y):

        # Find all unique classes
        self.classes = np.unique(y)

        for c in self.classes:

            # Select samples belonging to this class
            X_class = X[y == c]

            # Number of samples in this class
            class_count = len(X_class)

            # Prior probability
            self.priors[c] = class_count / len(X)

            # Store probabilities for this class
            self.feature_probabilities[c] = []

            # Calculate probability for every feature
            for feature_index in range(X.shape[1]):

                # Count how many times feature = 1
                count_one = np.sum(
                    X_class[:, feature_index] == 1
                )

                # Laplace smoothing
                probability_one = (
                    count_one + 1
                ) / (
                    class_count + 2
                )

                self.feature_probabilities[c].append(
                    probability_one
                )

        return self

    def predict_one(self, x):

        probabilities = {}

        for c in self.classes:

            # Start with prior probability
            probability = self.priors[c]

            for feature_index in range(len(x)):

                probability_one = (
                    self.feature_probabilities[c][feature_index]
                )

                if x[feature_index] == 1:

                    likelihood = probability_one

                else:

                    likelihood = 1 - probability_one

                probability *= likelihood

            probabilities[c] = probability

        return max(
            probabilities,
            key=probabilities.get
        )

    def predict(self, X):

        predictions = []

        for x in X:

            predictions.append(
                self.predict_one(x)
            )

        return np.array(predictions)

    def accuracy(self, X, y):

        predictions = self.predict(X)

        return np.mean(predictions == y)