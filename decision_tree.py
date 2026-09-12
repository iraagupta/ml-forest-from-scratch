import numpy as np

class Node:
    # literally just a box in our flowchart
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature       # which column are we judging rn
        self.threshold = threshold   # the cutoff point (like is age <= 18)
        self.left = left             # path for yes
        self.right = right           # path for no
        self.value = value           # if it's the end of the line, what's the final answer

    def is_leaf_node(self):
        # checking if we actually have an answer yet
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split  # min data points needed to keep splitting
        self.max_depth = max_depth                  # stopping it from overthinking
        self.n_features = n_features                # how many columns we randomly pick
        self.root = None

    def _gini(self, y):
        # calculating how "mixed up" the data is (0 means pure, perfect split)
        if len(y) == 0:
            return 0.0 # dodging a zero division error bc math is hard 😭
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1.0 - np.sum(ps ** 2)

    def fit(self, X, y):
        # figuring out how many columns we're dealing with
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # time to stop growing if it hit the limit or everyone is the same category
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # pick random columns so the trees don't all look identical
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # recursively splitting... basically tree-ception
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        # testing literally every single cutoff point to find the least messy split
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_gini = self._gini(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculating the vibe shift after the split
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l, gini_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r

        return parent_gini - child_gini

    def _split(self, X_column, split_thresh):
        # literally just grouping them into yes and no buckets
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0 # literally just guessing class 0 if the leaf is empty so we don't crash 😭
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # walking down the flowchart until we hit a leaf
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)