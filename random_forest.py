import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees                      # like how many trees are we spawning in this squad
        self.max_depth = max_depth                  # max questions a tree can ask before it needs to stfu
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            # bootstrapping basically giving every tree a slightly diff dataset so they don't just copy each other 💀
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        # picking random rows but like with replacement so some rows get picked twice and some get ghosted 😭
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        # if there's a tie in the votes we just pick randomly bc im not dealing with that drama
        counts = np.bincount(y)
        max_count = np.max(counts)
        candidates = np.where(counts == max_count)[0]
        return np.random.choice(candidates)

    def predict(self, X):
        # asking every single tree for their vibe check / guess
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # majority wins basically
        y_pred = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)