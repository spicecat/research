from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_diabetes
from sklearn.utils import resample
import numpy as np
import pandas as pd


def bootstrap_data(X, y, k):
    # Create a dataframe to store the bootstrap predictions
    bootstrap_preds = pd.DataFrame()

    # Generate k bootstrap samples and train a decision tree on each
    for i in range(k):
        X_resample, y_resample = resample(X, y)
        tree = DecisionTreeClassifier()
        tree.fit(X_resample, y_resample)
        preds = tree.predict(X)
        bootstrap_preds[f'bootstrap_{i}'] = preds

    # Append the bootstrap predictions to the original data
    X_extended = np.hstack((X, bootstrap_preds.values))

    return X_extended


def train_nn(X, y, hidden_nodes_arr):
    # Train a neural network on the extended data
    nn = MLPClassifier(hidden_layer_sizes=hidden_nodes_arr, max_iter=1000)
    nn.fit(X, y)

    return nn


def bootstrap_nn(X, y, k, hidden_nodes_arr):
    X_extended = bootstrap_data(X, y, k)
    print(X_extended)
    nn = train_nn(X_extended, y, hidden_nodes_arr)
    return nn


# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Train and test the function
nn = bootstrap_nn(X, y, k=5, hidden_nodes_arr=[10])
print("Training score:", nn.score(X, y))
