import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTree(max_depth=5)
tree.fit(X_train, y_train)

predictions = tree.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)



print("-" * 40)
print("First 3 flowers' measurements:\n", X[:3])
print("First 3 flowers' answers:", y[:3])
print(f"🚀 Day 1 Decision Tree Accuracy: {acc * 100:.2f}%")
print(f"Real labels:      {y_test}")
print(f"Tree predictions: {predictions}")
print("-" * 40)