import numpy as np
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
from graphviz import Digraph
from sklearn.tree import export_text, plot_tree


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, *, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases for the hidden layer
        self.weights_hidden = np.random.randn(input_dim, hidden_dim)
        self.bias_hidden = np.zeros(hidden_dim)

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.randn(hidden_dim, output_dim)
        self.bias_output = np.zeros(output_dim)

    def _forward(self, X):
        # Compute hidden layer activations
        self.z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.a_hidden = np.tanh(self.z_hidden)  # Tanh activation

        # Compute output layer activations
        self.z_output = np.dot(
            self.a_hidden, self.weights_output) + self.bias_output
        return self.z_output  # Linear output

    def _backward(self, X, y, output):
        # Compute the error between the output and the true labels
        output_error = output - y.reshape(-1, 1)

        # Compute gradients for the weights and biases of the output layer
        d_weights_output = np.dot(self.a_hidden.T, output_error)
        d_bias_output = np.mean(output_error, axis=0)

        # Compute hidden layer error and gradients
        hidden_error = np.dot(output_error, self.weights_output.T) * \
            (1 - self.a_hidden ** 2)  # Tanh derivative
        d_weights_hidden = np.dot(X.T, hidden_error)
        d_bias_hidden = np.mean(hidden_error, axis=0)

        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0
        d_weights_output = np.clip(
            d_weights_output, -max_grad_norm, max_grad_norm)
        d_bias_output = np.clip(d_bias_output, -max_grad_norm, max_grad_norm)
        d_weights_hidden = np.clip(
            d_weights_hidden, -max_grad_norm, max_grad_norm)
        d_bias_hidden = np.clip(d_bias_hidden, -max_grad_norm, max_grad_norm)

        # Update weights and biases using gradient descent
        self.weights_output -= self.learning_rate * d_weights_output
        self.bias_output -= self.learning_rate * d_bias_output
        self.weights_hidden -= self.learning_rate * d_weights_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden

    def fit(self, X, y):
        for epoch in range(self.epochs):
            output = self._forward(X)
            self._backward(X, y, output)
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)
            # if epoch % 200 == 0:
            #     print(
            #         f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self._forward(X)

    def get_weights(self):
        return self.weights_output

    def visualize(self):
        dot = Digraph()

        # Add input layer nodes
        for i in range(self.input_dim):
            dot.node(f'X{i}', f'X{i}')

        # Add hidden layer nodes
        for i in range(self.hidden_dim):
            dot.node(f'H{i}', f'H{i}')

        # Add output layer nodes
        dot.node('Y', 'Output')

        # Add edges from input to hidden layer
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                dot.edge(f'X{i}', f'H{j}')

        # Add edges from hidden layer to output
        for i in range(self.hidden_dim):
            dot.edge(f'H{i}', 'Y')

        return dot


class Ensemble:
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = [DecisionTreeRegressor(
            max_depth=5, random_state=i) for i in range(num_trees)]

    def __iter__(self):
        return iter(self.trees)

    def fit(self, X, y):
        for tree in self.trees:
            tree.fit(X, y)

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
        # Predict using the trees in the hidden layer
        # hidden_activations = np.tanh(np.dot(X, self.weights1) + self.bias1)
        # tree_predictions = np.column_stack(
        #     [tree.predict(hidden_activations) for tree in self.trees_hidden])

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
        # return np.mean([tree.predict(X) for tree in self.trees], axis=0)


# FONN1: Custom MLP with trees in the input layer


class FONN1(MLP, Ensemble):
    def __init__(self, input_dim, hidden_dim, output_dim, num_trees_input, **kwargs):
        MLP.__init__(self, input_dim + num_trees_input,
                     hidden_dim, output_dim, **kwargs)
        self.trees = Ensemble(num_trees_input)

    def _forward(self, X):
        # Generate tree outputs for the input layer
        input_tree_outputs = np.column_stack(
            [tree.predict(X) for tree in self.trees])
        combined_input = np.hstack((X, input_tree_outputs))
        return super()._forward(combined_input)

    def _backward(self, X, y, output):
        input_tree_outputs = np.column_stack(
            [tree.predict(X) for tree in self.trees])
        combined_input = np.hstack((X, input_tree_outputs))
        super()._backward(combined_input, y, output)

    def fit(self, X, y):
        self.trees.fit(X, y)
        MLP.fit(self, X, y)

# FONN2: Custom MLP with trees in hidden layer


class FONN2(MLP, Ensemble):
    def __init__(self, input_dim, hidden_dim, output_dim, num_trees_hidden, **kwargs):
        MLP.__init__(self, input_dim, hidden_dim, output_dim, **kwargs)
        self.trees = Ensemble(num_trees_hidden)
        self.weights_output = np.random.randn(
            hidden_dim + num_trees_hidden, output_dim)

    def _forward(self, X):
        # Compute hidden layer activations
        self.z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.a_hidden = np.tanh(self.z_hidden)  # Tanh activation

        # Generate tree outputs for the hidden layer
        hidden_tree_outputs = np.column_stack(
            [tree.predict(X) for tree in self.trees])
        self.combined_hidden = np.hstack((self.a_hidden, hidden_tree_outputs))

        # Compute output layer activations
        self.z_output = np.dot(self.combined_hidden,
                               self.weights_output) + self.bias_output
        return self.z_output  # Linear output

    def _backward(self, X, y, output):
        # Compute the error between the output and the true labels
        output_error = output - y.reshape(-1, 1)

        # Compute gradients for the weights and biases of the output layer
        d_weights_output = np.dot(self.combined_hidden.T, output_error)
        d_bias_output = np.mean(output_error, axis=0)

        # Compute hidden layer error and gradients
        hidden_error = np.dot(
            output_error, self.weights_output[:self.hidden_dim].T) * (1 - self.a_hidden ** 2)  # Tanh derivative
        d_weights_hidden = np.dot(X.T, hidden_error)
        d_bias_hidden = np.mean(hidden_error, axis=0)

        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0
        d_weights_output = np.clip(
            d_weights_output, -max_grad_norm, max_grad_norm)
        d_bias_output = np.clip(d_bias_output, -max_grad_norm, max_grad_norm)
        d_weights_hidden = np.clip(
            d_weights_hidden, -max_grad_norm, max_grad_norm)
        d_bias_hidden = np.clip(d_bias_hidden, -max_grad_norm, max_grad_norm)

        # Update weights and biases using gradient descent
        self.weights_output -= self.learning_rate * d_weights_output
        self.bias_output -= self.learning_rate * d_bias_output
        self.weights_hidden -= self.learning_rate * d_weights_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden

    def fit(self, X, y):
        self.trees.fit(X, y)
        MLP.fit(self, X, y)

# TREENN1: Custom MLP with 1 tree in the input layer


class TREENN1(FONN1):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, 1, **kwargs)

# TREENN2: Custom MLP with 1 tree in hidden layer


class TREENN2(FONN2):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(input_dim, hidden_dim, output_dim, 1, **kwargs)
