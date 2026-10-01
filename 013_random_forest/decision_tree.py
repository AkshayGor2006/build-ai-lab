import numpy as np

class Node:

    def __init__(
            self,
            feature_index=None,
            threshold=None,
            left=None,
            right=None,
            value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def __repr__(self):
        if self.value is not None:
            return f"Leaf(value={self.value})"

        return (
            f"Node(feature={self.feature_index}, "
            f"threshold={self.threshold}, "
            f"left={self.left}, "
            f"right={self.right})"
        )


class DecisionTreeClassifier:

    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def majority_class(self, y):
        counts = np.bincount(y.astype(int))
        return np.argmax(counts)

    # your other methods...

    def gini_impurity(self, y):

        classes, counts = np.unique(y, return_counts=True)

        probabilities = counts / len(y)

        gini = 1 - np.sum(probabilities ** 2)

        return gini

    def split(self, X, y, feature_index, threshold):

        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]

        return X_left, y_left, X_right, y_right

    def split_gini(self, y_left, y_right):

        total = len(y_left) + len(y_right)

        gini_left = self.gini_impurity(y_left)
        gini_right = self.gini_impurity(y_right)

        weighted_gini = (
            (len(y_left) / total) * gini_left
            +
            (len(y_right) / total) * gini_right
        )

        return weighted_gini

    def find_best_split(self, X, y):

        best_feature = None
        best_threshold = None
        best_gini = float("inf")

        for feature_index in range(X.shape[1]):

            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:

                X_left, y_left, X_right, y_right = self.split(
                    X,
                    y,
                    feature_index,
                    threshold
                )

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini = self.split_gini(y_left, y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold, best_gini

    def build_tree(self, X, y, depth=0):

        # Stop if node is pure
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        # Stop if maximum depth reached
        if depth >= self.max_depth:
            return Node(value=self.majority_class(y))

        # Find best split
        feature_index, threshold, gini = self.find_best_split(X, y)

        # Stop if no valid split exists
        if feature_index is None:
            return Node(value=self.majority_class(y))

        # Split the data
        X_left, y_left, X_right, y_right = self.split(
            X,
            y,
            feature_index,
            threshold
        )

        # Recursively build children
        left_child = self.build_tree(
            X_left,
            y_left,
            depth + 1
        )

        right_child = self.build_tree(
            X_right,
            y_right,
            depth + 1
        )

        # Create current decision node
        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def fit(self, X, y):

        self.root = self.build_tree(X, y)

        return self

    def predict_one(self, x, node):

        #we reached a leaf
        if node.value is not None:
            return node.value

        #decide which branch to follow 
        if x[node.feature_index] <= node.threshold:
            return self.predict_one(x, node.left)

        else:
            return self.predict_one(x, node.right)

    def predict(self, X):

        predictions = []

        for x in X:
            prediction = self.predict_one(x, self.root)
            predictions.append(prediction)

        return np.array(predictions)

    def accuracy(self, X, y):
        predictions = self.predict(X)

        correct = np.sum(predictions == y)

        return correct / len(y)