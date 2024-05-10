from tree.regression_tree import bootstrap_predictions
import numpy as np
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target  # type: ignore

X_train, y_train = X, y

# Test pruning
# best_alpha = k_fold_cv(X_train, y_train)
# tree = split(X_train, y_train)
# pruned_tree = prune_tree(tree, X_train, y_train, best_alpha)
# print(pruned_tree)

# Test bootstrap
reps = bootstrap_predictions(X_train, y_train)
X_extended = np.concatenate((X_train, reps), axis=1)
print(X_extended)
