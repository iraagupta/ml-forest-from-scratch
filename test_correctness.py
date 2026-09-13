import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnTree
from decision_tree import DecisionTree
# FIX: locking the seed so our custom tree doesn't change answers every run 😭
np.random.seed(42)

# grabbing a slightly harder dataset bc we are brave today
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== THE ULTIMATE SHOWDOWN ===")

# our homemade tree 💅
our_tree = DecisionTree(max_depth=5)
our_tree.fit(X_train, y_train)
our_preds = our_tree.predict(X_test)  # <-- this is the line that went missing 💀
our_acc = np.sum(our_preds == y_test) / len(y_test)
print(f"💅 our custom tree acc: {our_acc:.4f}")

# the corporate sklearn tree 👔 (testing correctly)
their_tree = SklearnTree(max_depth=5, random_state=42)
their_tree.fit(X_train, y_train)
their_preds = their_tree.predict(X_test)
their_acc = np.sum(their_preds == y_test) / len(y_test)
print(f"👔 sklearn's tree acc:  {their_acc:.4f}")

print("=============================\n")

# if we are within 5% of sklearn, we take the W
if our_acc >= their_acc - 0.05:
    print("✅ STATUS: PASS! our math is solid")
else:
    print("❌ STATUS: FAIL! back to the drawing board")