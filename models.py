import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import gen_batches, shuffle

import matplotlib.pyplot as plt
# from graphviz import Digraph
from sklearn.tree import export_text, plot_tree

ACTIVATIONS = {
    'relu': lambda x: np.maximum(0, x),
    'tanh': lambda x: np.tanh(x)
}
DERIVATIVES = {
    'relu': lambda x: (x > 0).astype(float),
    'tanh': lambda x: 1 - np.tanh(x)**2
}


class Ensemble(BaseEstimator):
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = [DecisionTreeRegressor(
            max_depth=5, random_state=i) for i in range(num_trees)]

    def __iter__(self):
        return iter(self.trees)

    def fit(self, X, y):
        for tree in self.trees:
            tree.fit(X, y)
        self.is_fitted_ = True

    def get_tree_importances(self):
        importances = []
        for i, tree in enumerate(self):
            importances.append(tree.feature_importances_)
            print(f"Tree {i} feature importances:\n{tree.feature_importances_}")
            tree_rules = export_text(tree)
            print(f"Tree {i} structure:\n{tree_rules}")
            plt.figure(figsize=(20, 10))
            plot_tree(tree, filled=True)
            plt.title(f"Tree {i} Visualization")
            plt.show()
        return importances

    def predict(self, X):
        # Predict using the trees
        tree_predictions = np.column_stack(
            [tree.predict(X) for tree in self])

        # Compute feature importance weights for each tree
        tree_importances = [
            tree.feature_importances_ for tree in self]
        # Normalize the importances to sum to 1 for each tree
        tree_weights = np.array([importances / np.sum(importances) if np.sum(
            importances) > 0 else np.ones_like(importances) for importances in tree_importances])
        # Average the normalized importances to get the final weights for each tree
        final_weights = np.mean(tree_weights, axis=1)

        # Compute the weighted average of the tree predictions
        weighted_tree_predictions = np.average(
            tree_predictions, axis=1, weights=final_weights)
        return weighted_tree_predictions

    def score(self, X, y):
        return r2_score(y, self.predict(X))


class MLP(BaseEstimator):
    def __init__(self, input_dim, hidden_dim, output_dim, *, activation='relu', batch_size=200, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_curve_ = []
        self.coefs_ = []
        self.intercepts_ = []
        self.coef_grads = []
        self.intercept_grads = []
        self._initialize()

    def _initialize(self):
        hidden_dim = list(self.hidden_dim) if hasattr(
            self.hidden_dim, "__iter__") else [self.hidden_dim]
        layer_units = [self.input_dim, *hidden_dim, self.output_dim]
        self.n_layers_ = len(layer_units)

        for i in range(self.n_layers_ - 1):
            coef_init = np.random.randn(layer_units[i], layer_units[i + 1])
            intercept_init = np.zeros(layer_units[i+1])
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        self.coef_grads = [
            np.empty((n_fan_in_, n_fan_out_))
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]
        self.intercept_grads = [
            np.empty(n_fan_out_) for n_fan_out_ in layer_units[1:]
        ]

    def _forward(self, X):
        activations = [X] * self.n_layers_
        # Compute hidden layer activations
        for i in range(self.n_layers_-1):
            activations[i+1] = np.dot(activations[i], self.coefs_[i])
            activations[i+1] += self.intercepts_[i]
            if i+1 != self.n_layers_-1:
                # Activation for hidden layers
                activations[i + 1] = ACTIVATIONS[self.activation](
                    activations[i+1]
                )

        return activations

    def _backward(self, y, activations):
        loss = activations[-1] - y.reshape(-1, 1)
        # Compute the gradients for the hidden layers
        for i in range(self.n_layers_ - 1, 0, -1):
            if i != self.n_layers_ - 1:
                loss = np.dot(loss, self.coefs_[i].T) * DERIVATIVES[self.activation](
                    activations[i]
                )
            self.coef_grads[i - 1] = np.dot(
                activations[i-1].T, loss) / y.shape[0]
            self.intercept_grads[i-1] = np.mean(loss, axis=0)

        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0
        for i in range(self.n_layers_-1):
            self.coef_grads[i] = np.clip(
                self.coef_grads[i], -max_grad_norm, max_grad_norm)
            self.intercept_grads[i] = np.clip(
                self.intercept_grads[i], -max_grad_norm, max_grad_norm)
            # Update weights and biases using gradient descent
            self.coefs_[i] -= self.learning_rate * self.coef_grads[i]
            self.intercepts_[i] -= self.learning_rate * self.intercept_grads[i]

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.reshape(-1, 1)
        for epoch in range(self.epochs):
            X, y = shuffle(X, y)  # type: ignore
            batches = gen_batches(len(y), self.batch_size)
            for batch in batches:
                activations = self._forward(X[batch])
                self._backward(y[batch], activations)

            loss = np.mean((y - self._forward(X)[-1]) ** 2)
            self.loss_curve_.append(loss)

            # if epoch % 200 == 0:
            #     loss = np.mean((y - self._forward(X)) ** 2)
            #     print(f'Epoch {epoch}, Loss: {loss}')
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self._forward(X)[-1]

    def score(self, X, y):
        return r2_score(y, self.predict(X))

    # def visualize(self):
    #     dot = Digraph()

    #     # Add input layer nodes
    #     for i in range(self.input_dim):
    #         dot.node(f'X{i}', f'X{i}')

    #     # Add hidden layer nodes
    #     for i in range(self.hidden_dim):
    #         dot.node(f'H{i}', f'H{i}')

    #     # Add output layer nodes
    #     dot.node('Y', 'Output')

    #     # Add edges from input to hidden layer
    #     for i in range(self.input_dim):
    #         for j in range(self.hidden_dim):
    #             dot.edge(f'X{i}', f'H{j}')

    #     # Add edges from hidden layer to output
    #     for i in range(self.hidden_dim):
    #         dot.edge(f'H{i}', 'Y')

    #     return dot

# FONN1: Custom MLP with trees in the input layer


class FONN1(MLP, Ensemble):
    def __init__(self, input_dim, hidden_dim, output_dim, num_trees_input, *, activation='relu', batch_size=200, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_trees_input = num_trees_input
        self.activation = activation
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_curve_ = []
        self.coefs_ = []
        self.intercepts_ = []
        self.coef_grads = []
        self.intercept_grads = []
        self._initialize()
        self.trees = Ensemble(num_trees_input)

    def _initialize(self):
        hidden_dim = list(self.hidden_dim) if hasattr(
            self.hidden_dim, "__iter__") else [self.hidden_dim]
        layer_units = [self.input_dim+self.num_trees_input,
                       *hidden_dim, self.output_dim]
        self.n_layers_ = len(layer_units)

        for i in range(self.n_layers_ - 1):
            coef_init = np.random.randn(layer_units[i], layer_units[i + 1])
            intercept_init = np.zeros(layer_units[i+1])
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        self.coef_grads = [
            np.empty((n_fan_in_, n_fan_out_))
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]
        self.intercept_grads = [
            np.empty(n_fan_out_) for n_fan_out_ in layer_units[1:]
        ]

    def _forward(self, X):
        # Generate tree outputs for the input layer
        input_tree_outputs = np.column_stack(
            [tree.predict(X) for tree in self.trees])
        combined_input = np.hstack((X, input_tree_outputs))
        return super()._forward(combined_input)

    def fit(self, X, y):
        self.trees.fit(X, y)
        MLP.fit(self, X, y)

    def get_params(self, deep):
        params = super().get_params(deep)
        params.update({
            'num_trees_input': self.trees.num_trees
        })
        return params

# FONN2: Custom MLP with trees in hidden layer


class FONN2(MLP, Ensemble):
    def __init__(self, input_dim, hidden_dim, output_dim, num_trees_hidden, *, activation='relu', batch_size=200, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_trees_hidden = num_trees_hidden
        self.activation = activation
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_curve_ = []
        self.coefs_ = []
        self.intercepts_ = []
        self.coef_grads = []
        self.intercept_grads = []
        self._initialize()
        self.trees = Ensemble(num_trees_hidden)

    def _initialize(self):
        hidden_dim = list(self.hidden_dim) if hasattr(
            self.hidden_dim, "__iter__") else [self.hidden_dim]
        hidden_dim[-1] += self.num_trees_hidden
        layer_units = [self.input_dim, *hidden_dim, self.output_dim]
        self.n_layers_ = len(layer_units)

        for i in range(self.n_layers_ - 1):
            coef_init = np.random.randn(layer_units[i], layer_units[i + 1])
            intercept_init = np.zeros(layer_units[i+1])
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)

        self.coef_grads = [
            np.empty((n_fan_in_, n_fan_out_))
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]
        self.intercept_grads = [
            np.empty(n_fan_out_) for n_fan_out_ in layer_units[1:]
        ]

    def _forward(self, X):
        activations = [X] * self.n_layers_
        # Compute hidden layer activations
        for i in range(self.n_layers_-1):
            if i+1 == self.n_layers_-1:
                hidden_tree_outputs = np.column_stack(
                    [tree.predict(X) for tree in self.trees])
                activations[i][:, -
                               self.num_trees_hidden:] = hidden_tree_outputs
            activations[i+1] = np.dot(activations[i], self.coefs_[i])
            activations[i+1] += self.intercepts_[i]
            if i+1 != self.n_layers_-1:
                # Activation for hidden layers
                activations[i + 1] = ACTIVATIONS[self.activation](
                    activations[i+1]
                )

        return activations

    def _backward(self, y, activations):
        loss = activations[-1] - y.reshape(-1, 1)
        # Compute the gradients for the hidden layers
        for i in range(self.n_layers_ - 1, 0, -1):
            if i != self.n_layers_ - 1:
                loss = np.dot(loss, self.coefs_[i].T) * DERIVATIVES[self.activation](
                    activations[i]
                )
            self.coef_grads[i - 1] = np.dot(
                activations[i-1].T, loss) / y.shape[0]
            self.intercept_grads[i-1] = np.mean(loss, axis=0)

        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0
        for i in range(self.n_layers_-1):
            self.coef_grads[i] = np.clip(
                self.coef_grads[i], -max_grad_norm, max_grad_norm)
            self.intercept_grads[i] = np.clip(
                self.intercept_grads[i], -max_grad_norm, max_grad_norm)
            # Update weights and biases using gradient descent
            self.coefs_[i] -= self.learning_rate * self.coef_grads[i]
            self.intercepts_[i] -= self.learning_rate * self.intercept_grads[i]

    def fit(self, X, y):
        self.trees.fit(X, y)
        MLP.fit(self, X, y)

    def get_params(self, deep):
        params = super().get_params(deep)
        params.update({
            'num_trees_hidden': self.trees.num_trees
        })
        return params

# TREENN1: Custom MLP with 1 tree in the input layer


class TREENN1(FONN1):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, num_trees_input=1, **kwargs)

# TREENN2: Custom MLP with 1 tree in hidden layer


class TREENN2(FONN2):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, num_trees_hidden=1, **kwargs)
