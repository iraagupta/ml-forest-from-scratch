import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
from random_forest import RandomForest

# 1. making a super messy dataset on purpose to prove a point 💅
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5, 
    n_redundant=2, n_clusters_per_class=2, flip_y=0.2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print("\n=== RUNNING THE OVERFITTING DRAMA ===")

# model 1: no boundaries tree (will literally memorize the training data and bomb the test 💀)
tree_unrestricted = DecisionTree(max_depth=100)
tree_unrestricted.fit(X_train, y_train)
print(f"1. unhinged tree     -> train acc: {accuracy(y_train, tree_unrestricted.predict(X_train)):.4f} | test acc: {accuracy(y_test, tree_unrestricted.predict(X_test)):.4f}")

# model 2: pruned tree (we actually gave it boundaries)
tree_pruned = DecisionTree(max_depth=3)
tree_pruned.fit(X_train, y_train)
print(f"2. regularized tree  -> train acc: {accuracy(y_train, tree_pruned.predict(X_train)):.4f} | test acc: {accuracy(y_test, tree_pruned.predict(X_test)):.4f}")

# model 3: random forest squad coming in to fix everything
forest = RandomForest(n_trees=15, max_depth=10)
forest.fit(X_train, y_train)
print(f"3. random forest     -> train acc: {accuracy(y_train, forest.predict(X_train)):.4f} | test acc: {accuracy(y_test, forest.predict(X_test)):.4f}")
print("======================================\n")