import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from typing import Literal

import matplotlib.pyplot as plt


class CustomMLP(MLPRegressor):
    def plot_loss(self, title='Loss Curve'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_curve_)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid()
        plt.show()


class FONN1(CustomMLP):
    def __init__(self, num_trees=10, hidden_layer_sizes=(100,), activation: Literal['relu', 'identity', 'logistic', 'tanh'] = 'relu',  **kwargs):
        super().__init__(hidden_layer_sizes, activation, **kwargs)
        self.num_trees = num_trees
        self.ensemble = RandomForestRegressor(num_trees)

    def _concat_tree(self, X):
        return np.hstack((X, np.column_stack([e.predict(X) for e in self.ensemble.estimators_])))

    def fit(self, X: np.ndarray, y):
        self.ensemble.fit(X, y)
        return super().fit(self._concat_tree(X), y)

    def predict(self, X: np.ndarray):
        return super().predict(self._concat_tree(X))


class TREENN1(FONN1):
    def __init__(self, hidden_layer_sizes=(100,), activation: Literal['relu', 'identity', 'logistic', 'tanh'] = 'relu',  **kwargs):
        super().__init__(1, hidden_layer_sizes, activation, **kwargs)


class FONN2(CustomMLP):
    def __init__(self, num_trees=10, hidden_layer_sizes=(100,), activation: Literal['relu', 'identity', 'logistic', 'tanh'] = 'relu',  **kwargs):
        super().__init__(hidden_layer_sizes, activation, **kwargs)
        self.num_trees = num_trees
        self.ensemble = RandomForestRegressor(num_trees)

    def _concat_tree(self, activations):
        tree_outputs = np.column_stack(
            [e.predict(activations[0]) for e in self.ensemble.estimators_])
        activations[-2] = np.hstack((tree_outputs,
                                    activations[-2][:, self.num_trees:]))
        return activations

    def _forward_pass(self, activations):
        activations = super()._forward_pass(activations)
        activations = self._concat_tree(activations)
        return activations

    def fit(self, X, y):
        self.ensemble.fit(X, y)
        return super().fit(X, y)


class TREENN2(FONN2):
    def __init__(self, hidden_layer_sizes=(100,), activation: Literal['relu', 'identity', 'logistic', 'tanh'] = 'relu',  **kwargs):
        super().__init__(1, hidden_layer_sizes, activation, **kwargs)


if __name__ == '__main__':
    import pandas as pd
    import time
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    import numpy as np

    np.random.seed(42)

    # Load the Boston dataset
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22,  # type: ignore
                         header=None)  # type: ignore
    X = np.hstack([raw_df.values[::2, :-1], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    def train_evaluate_model(model, X_train, X_test, y_train, y_test):
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        train_time = end_time - start_time

        start_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()
        comp_time = end_time - start_time

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

        return r2, mae, mse, train_time, comp_time

    print(X_train.shape, X_test.shape)

    early_stopping=False

    mlp = CustomMLP(10, max_iter=10000, early_stopping=early_stopping)
    fonn1 = FONN1(5, (10,), max_iter=10000, early_stopping=early_stopping)
    fonn2 = FONN2(5, (15,), max_iter=10000, early_stopping=early_stopping)
    treenn1 = TREENN1((10,), max_iter=10000, early_stopping=early_stopping)
    treenn2 = TREENN2((10,), max_iter=10000, early_stopping=early_stopping)

    print(train_evaluate_model(mlp, X_train, X_test, y_train, y_test))
    print(train_evaluate_model(fonn1, X_train, X_test, y_train, y_test))
    print(train_evaluate_model(fonn2, X_train, X_test, y_train, y_test))
    print(train_evaluate_model(treenn1, X_train, X_test, y_train, y_test))
    print(train_evaluate_model(treenn2, X_train, X_test, y_train, y_test))

    mlp.plot_loss('MLP')
    fonn1.plot_loss('FONN1')
    fonn2.plot_loss('FONN2')
    treenn1.plot_loss('TREENN1')
    treenn2.plot_loss('TREENN2')
