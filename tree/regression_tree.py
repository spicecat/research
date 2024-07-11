from sklearn.utils import resample
import numpy as np
from sklearn.model_selection import KFold


def mse(y_true, y_pred):
    """
    Calculates the mean squared error between true and predicted values.
    """
    return np.mean((y_true - y_pred) ** 2)


class Node:
    """
    Represents a node in the regression tree.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def __repr__(self):
        if self.value is not None:
            return f"Node(value={self.value:.2f})"
        else:
            return f"Node(feature={self.feature}, threshold={self.threshold:.2f}, left={self.left}, right={self.right})"

    def __str__(self, depth=0):
        if self.value is not None:
            return "  " * depth + f"{depth} value={self.value:.2f}"
        else:
            lines = [
                "  " * depth + f"{depth} [{self.feature} < {self.threshold:.2f}]"]
            if self.left:
                lines.append(self.left.__str__(depth + 1))
            if self.right:
                lines.append(self.right.__str__(depth + 1))
            return "\n".join(lines)

    def predict(self, X):
        """
        Predicts the target values for the given input data using the regression tree.
        """
        if self.value is not None or self.left is None or self.right is None:
            return self.value
        if X[self.feature] < self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)

    def summary(self, X, y):
        """
        Provides a summary of the regression tree.
        """
        variables_used = set()
        n_terminal_nodes = 0
        total_error = 0
        samples = len(y)
        sum_squared_residuals = 0

        def traverse(node):
            nonlocal n_terminal_nodes, total_error, sum_squared_residuals

            if node.value is not None:
                n_terminal_nodes += 1
                total_error += mse(y, np.full_like(y, node.value))
                sum_squared_residuals += np.sum((y - node.value) ** 2)
                return

            variables_used.add(node.feature)
            left_idx = X[:, node.feature] < node.threshold
            right_idx = ~left_idx

            left_predictions = [predict(x, node.left) for x in X[left_idx]]
            right_predictions = [predict(x, node.right) for x in X[right_idx]]

            total_error += mse(y[left_idx], left_predictions)
            total_error += mse(y[right_idx], right_predictions)
            sum_squared_residuals += np.sum(
                (y[left_idx] - left_predictions) ** 2)
            sum_squared_residuals += np.sum(
                (y[right_idx] - right_predictions) ** 2)

            traverse(node.left)
            traverse(node.right)

        traverse(self)
        residual_mean_deviance = total_error / \
            n_terminal_nodes if n_terminal_nodes else 0

        return {
            "Variables Used": variables_used,
            "Number of Terminal Nodes": n_terminal_nodes,
            "Total Error": total_error,
            "Residual Mean Deviance": residual_mean_deviance,
            "Samples": samples,
            "Sum of Squared Residuals": sum_squared_residuals
        }


def split(X, y, depth=0, max_depth=float('inf')):
    """
    Recursively builds a regression tree.
    """
    if depth >= max_depth or len(y) <= 1:
        return Node(value=np.mean(y))

    best_mse = float('inf')
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        # Sort X by the current feature
        sorted_idx = np.argsort(X[:, feature])
        sorted_X = X[sorted_idx]
        sorted_y = y[sorted_idx]

        thresholds = np.unique(sorted_X[:, feature])
        for threshold in thresholds:
            left_idx = sorted_X[:, feature] < threshold
            right_idx = ~left_idx

            left_y = sorted_y[left_idx]
            right_y = sorted_y[right_idx]

            if len(left_y) == 0 or len(right_y) == 0:
                continue

            left_mse = mse(left_y, np.mean(left_y))
            right_mse = mse(right_y, np.mean(right_y))
            total_mse = (len(left_y) * left_mse +
                         len(right_y) * right_mse) / len(y)

            if total_mse < best_mse:
                best_mse = total_mse
                best_feature = feature
                best_threshold = threshold

    if best_feature is None:
        return Node(value=np.mean(y))

    left_idx = X[:, best_feature] < best_threshold
    right_idx = ~left_idx
    left_node = split(X[left_idx], y[left_idx], depth + 1, max_depth)
    right_node = split(X[right_idx], y[right_idx], depth + 1, max_depth)
    return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)


def predict(X, tree):
    """
    Predicts the target values for the given input data using the regression tree.
    """
    if tree.value is not None:
        return tree.value
    if X[tree.feature] < tree.threshold:
        return predict(X, tree.left)
    else:
        return predict(X, tree.right)


def prune_tree(tree, X, y, alpha):
    """
    Prunes the regression tree based on the given alpha value.
    """
    if tree.value is not None:
        return tree

    left_idx = X[:, tree.feature] < tree.threshold
    right_idx = ~left_idx

    tree.left = prune_tree(tree.left, X[left_idx], y[left_idx], alpha)
    tree.right = prune_tree(tree.right, X[right_idx], y[right_idx], alpha)

    if tree.left.value is not None and tree.right.value is not None:
        node_error = mse(y, [predict(x, tree) for x in X])
        leaf_error = mse(y, np.mean(y))
        if leaf_error <= node_error + alpha:
            return Node(value=np.mean(y))

    return tree


def k_fold_cv(X, y, k=5, max_depth=float('inf')):
    """
    Performs k-fold cross-validation to find the optimal pruned regression tree.
    """
    kf = KFold(n_splits=k)
    best_alpha = 0
    best_mse = float('inf')

    for alpha in np.logspace(-5, 10, 20):
        mse_scores = []
        for train_idx, test_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            tree = split(X_train, y_train, max_depth=max_depth)
            pruned_tree = prune_tree(tree, X_train, y_train, alpha)

            test_mse = mse(y_test, np.array(
                [predict(x, pruned_tree) for x in X_test]))
            mse_scores.append(test_mse)

        mean_mse = np.mean(mse_scores)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_alpha = alpha

    return best_alpha

def bootstrap_trees(X, y, R=5, k=5, max_depth=float('inf')):
    """
    Builds R regression trees using bootstrap samples of the data.
    """
    trees = []

    for i in range(R):
        # Generate a bootstrap sample
        X_sample, y_sample = resample(X, y) # type: ignore

        # Train a tree on the bootstrap sample and prune it
        best_alpha = k_fold_cv(X_sample, y_sample, k, max_depth)
        tree = split(X_sample, y_sample, max_depth=max_depth)
        pruned_tree = prune_tree(tree, X_sample, y_sample, best_alpha)
        trees.append(pruned_tree)
        print("sample:", i, pruned_tree.summary(X_sample, y_sample))

    return trees

def bootstrap_predictions(X, y, R=5, k=5, max_depth=float('inf')):
    """
    Adds R columns to X, each one filled with predictions from a pruned tree
    that was trained on a different bootstrap sample of the data.
    """
    # Initialize an array to hold the new columns
    new_columns = np.zeros((X.shape[0], R))
    
    trees = bootstrap_trees(X, y, R, k, max_depth)
    for i in range(R):
        new_columns[:, i] = np.array([predict(x, trees[i]) for x in X])

    return new_columns


if __name__ == "__main__":
    # Example Usage
    # Load the diabetes dataset
    from sklearn.datasets import load_diabetes
    from sklearn.utils import Bunch
    diabetes: Bunch = load_diabetes()  # type: ignore
    X, y = diabetes.data, diabetes.target
    X_train, y_train = X, y
    print(diabetes.feature_names)

    # Test pruning
    tree = split(X_train, y_train)
    print("Tree:", tree)
    print(tree.summary(X_train, y_train))
    best_alpha = k_fold_cv(X_train, y_train)
    print("Best Alpha:", best_alpha)
    pruned_tree = prune_tree(tree, X_train, y_train, best_alpha)
    print("Pruned Tree:", pruned_tree)
    print(pruned_tree.summary(X_train, y_train))

    # Test bootstrap
    reps = bootstrap_predictions(X_train, y_train)
    print("Reps:", reps.shape, reps)
    X_extended = np.concatenate((X_train, reps), axis=1)
    print(X_extended)
    print("Extended Shape:", X_extended.shape)
