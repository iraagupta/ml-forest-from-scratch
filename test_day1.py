import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree

# 1. Load sample dataset (Iris flowers)
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Instantiate and train your DecisionTree from scratch
tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

# 4. Predict on test set
predictions = tree.predict(X_test)

# 5. Calculate accuracy
acc = np.sum(predictions == y_test) / len(y_test)

print("-" * 40)
print(f"🚀 Day 1 Decision Tree Accuracy: {acc * 100:.2f}%")
print(f"Real labels:      {y_test}")
print(f"Tree predictions: {predictions}")
print("-" * 40)