# random forest from scratch 🌲

okay here is the breakdown of what i actually built and why single decision trees are lowkey tragic on their own.

## 1. the overfitting drama
if you let a single decision tree grow without setting boundaries (like `max_depth`), it gets way too obsessed with the training data. it literally memorizes every single row instead of learning the actual patterns. so when you give it new data on the final exam, it completely panics and fails. this is overfitting.

## 2. how random forests fix it (bagging)
a random forest fixes this by using a squad of trees instead of just one. it uses this concept called **bagging (bootstrap aggregating)**:
- **bootstrap:** every tree gets a slightly different, randomly shuffled version of the dataset (some rows get duplicated, some get ghosted). 
- **random features:** instead of letting the tree look at all the columns, we force it to only look at a random subset (the square root of the total columns) at each split.
- **aggregating:** finally, all the trees take a majority vote to make the final prediction.

bc every tree is slightly different and looking at different things, they balance each other out and stop the model from overfitting.

## 3. the chaotic bug history 
- **the zero division error:** tried to calculate gini impurity on an empty leaf and the math exploded. fixed it by returning 0.0.
- **the feature bug:** totally forgot that random forests are only supposed to look at the square root of features, not all of them. patched that in the `fit` function.
- **empty leaf crash:** our majority vote function tried to count empty arrays and crashed. added a fail-safe to just guess class 0.
- **the random seed leak:** our custom tree's accuracy kept randomly changing every run on the sklearn test. locked `np.random.seed(42)` to make it reproducible.