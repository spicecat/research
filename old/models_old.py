import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
from graphviz import Digraph
from sklearn.tree import export_text, plot_tree

# FONN1: Custom MLP with Trees in the input layer


class FONN1:
    def __init__(self, input_dim, hidden_dim, output_dim, num_trees_input):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_trees_input = num_trees_input

        # Initialize decision trees for the input layer
        self.trees_input = [DecisionTreeRegressor(
            max_depth=5, random_state=i) for i in range(num_trees_input)]

        # Initialize weights and biases for the hidden layer
        self.weights_hidden = np.random.randn(
            input_dim + num_trees_input, hidden_dim) * 0.01
        self.bias_hidden = np.zeros(hidden_dim)

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.randn(hidden_dim, output_dim) * 0.01
        self.bias_output = np.zeros(output_dim)

    def forward(self, X):
        input_tree_outputs = np.column_stack(
            [tree.predict(X) for tree in self.trees_input])
        combined_input = np.hstack((X, input_tree_outputs))

        # Compute hidden layer activations
        self.z_hidden = np.dot(
            combined_input, self.weights_hidden) + self.bias_hidden
        self.a_hidden = np.tanh(self.z_hidden)  # Tanh activation

        # Compute output layer activations
        self.z_output = np.dot(
            self.a_hidden, self.weights_output) + self.bias_output
        return self.z_output  # Linear output

    def backward(self, X, y, output, learning_rate_init):
        # Compute the error between the output and the true labels
        output_error = output - y.reshape(-1, 1)

        # Compute gradients for the weights and biases of the output layer
        d_weights_output = np.dot(self.a_hidden.T, output_error) / X.shape[0]
        d_bias_output = np.mean(output_error, axis=0)

        # Compute hidden layer error and gradients
        hidden_error = np.dot(output_error, self.weights_output.T) * \
            (1 - self.a_hidden ** 2)  # Tanh derivative
        input_tree_outputs = np.column_stack(
            [tree.predict(X) for tree in self.trees_input])
        combined_input = np.hstack((X, input_tree_outputs))
        d_weights_hidden = np.dot(combined_input.T, hidden_error) / X.shape[0]
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
        self.weights_output -= learning_rate_init * d_weights_output
        self.bias_output -= learning_rate_init * d_bias_output
        self.weights_hidden -= learning_rate_init * d_weights_hidden
        self.bias_hidden -= learning_rate_init * d_bias_hidden

    def fit(self, X, y, max_iter, learning_rate_init):
        # Generate tree outputs for the input layer
        for tree in self.trees_input:
            tree.fit(X, y)

        for epoch in range(max_iter):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate_init)
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)
            if epoch % 100 == 0:
                print(f'Epoch {iter}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)

    def visualize(self):
        dot = Digraph()

        # Add input layer nodes
        for i in range(self.input_dim):
            dot.node(f'X{i}', f'X{i}')

        # Add tree nodes
        for i in range(self.num_trees_input):
            dot.node(f'T{i}', f'T{i}')

        # Add hidden layer nodes
        for i in range(self.hidden_dim):
            dot.node(f'H{i}', f'H{i}')

        # Add output layer nodes
        dot.node('Y', 'Output')

        # Add edges from input to hidden layer
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                dot.edge(f'X{i}', f'H{j}')

        # Add edges from trees to hidden layer
        for i in range(self.num_trees_input):
            for j in range(self.hidden_dim):
                dot.edge(f'T{i}', f'H{j}')

        # Add edges from hidden layer to output
        for i in range(self.hidden_dim):
            dot.edge(f'H{i}', 'Y')

        return dot

    def get_tree_importances(self):
        importances = []
        for i, tree in enumerate(self.trees_input):
            importances.append(tree.feature_importances_)
            print(f"Tree {i} feature importances:\n{
                  tree.feature_importances_}")
            tree_rules = export_text(tree)
            print(f"Tree {i} structure:\n{tree_rules}")
            plt.figure(figsize=(20, 10))
            plot_tree(tree, filled=True)
            plt.title(f"Tree {i} Visualization")
            plt.show()
        return importances

# Custom MLP with 10 trees in hidden layer


class FONN2:
    def __init__(self, input_dim, hidden_dim, output_dim, num_trees_hidden):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_trees_hidden = num_trees_hidden

        # Initialize weights and biases for the first hidden layer
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.bias1 = np.zeros(hidden_dim)

        # Initialize weights and biases for the output layer
        self.weights2 = np.random.randn(
            hidden_dim + num_trees_hidden, output_dim) * 0.01
        self.bias2 = np.zeros(output_dim)

        # Initialize decision trees for the hidden layer
        self.trees_hidden = [DecisionTreeRegressor(
            max_depth=5, random_state=i) for i in range(num_trees_hidden)]

    def forward(self, X):
        # Compute hidden layer activations
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = np.tanh(self.z1)  # Tanh activation

        hidden_tree_outputs = np.column_stack(
            [tree.predict(X) for tree in self.trees_hidden])  # tree.predict(self.a1)
        self.combined_hidden = np.hstack((self.a1, hidden_tree_outputs))

        # Compute output layer activations
        self.z2 = np.dot(self.combined_hidden, self.weights2) + self.bias2
        return self.z2  # Linear output

    def backward(self, X, y, output, learning_rate_init):
        # Compute the error between the output and the true labels
        output_error = output - y.reshape(-1, 1)

        # Compute gradients for the weights and biases of the output layer
        d_weights2 = np.dot(self.combined_hidden.T, output_error) / X.shape[0]
        d_bias2 = np.mean(output_error, axis=0)

        # Compute hidden layer error and gradients
        hidden_error = np.dot(
            # Tanh derivative
            output_error, self.weights2[:self.hidden_dim].T) * (1 - self.a1 ** 2)
        d_weights1 = np.dot(X.T, hidden_error) / X.shape[0]
        d_bias1 = np.mean(hidden_error, axis=0)

        # Gradient clipping to prevent exploding gradients
        max_grad_norm = 1.0
        d_weights2 = np.clip(d_weights2, -max_grad_norm, max_grad_norm)
        d_bias2 = np.clip(d_bias2, -max_grad_norm, max_grad_norm)
        d_weights1 = np.clip(d_weights1, -max_grad_norm, max_grad_norm)
        d_bias1 = np.clip(d_bias1, -max_grad_norm, max_grad_norm)

        # Update weights and biases using gradient descent
        self.weights2 -= learning_rate_init * d_weights2
        self.bias2 -= learning_rate_init * d_bias2
        self.weights1 -= learning_rate_init * d_weights1
        self.bias1 -= learning_rate_init * d_bias1

    def fit(self, X, y, max_iter, learning_rate_init):
        # Generate tree outputs for the hidden layer
        for tree in self.trees_hidden:
            tree.fit(X, y)

        for epoch in range(max_iter):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate_init)
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)
            if epoch % 100 == 0:
                print(f'Epoch {iter}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)

    def visualize(self):
        dot = Digraph()

        # Add input layer nodes
        for i in range(self.input_dim):
            dot.node(f'X{i}', f'X{i}')

        # Add hidden layer nodes
        for i in range(self.hidden_dim):
            dot.node(f'H{i}', f'H{i}')

        # Add tree nodes
        for i in range(self.num_trees_hidden):
            dot.node(f'T{i}', f'T{i}')

        # Add output layer nodes
        dot.node('Y', 'Output')

        # Add edges from input to hidden layer
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                dot.edge(f'X{i}', f'H{j}')

        # Add edges from hidden layer to output
        for i in range(self.hidden_dim):
            dot.edge(f'H{i}', 'Y')

        # Add edges from trees to output
        for i in range(self.num_trees_hidden):
            dot.edge(f'T{i}', 'Y')

        return dot

    def get_tree_importances(self):
        importances = []
        for i, tree in enumerate(self.trees_hidden):
            importances.append(tree.feature_importances_)
            print(f"Tree {i} feature importances:\n{
                  tree.feature_importances_}")
            tree_rules = export_text(tree)
            print(f"Tree {i} structure:\n{tree_rules}")
            plt.figure(figsize=(20, 10))
            plot_tree(tree, filled=True)
            plt.title(f"Tree {i} Visualization")
            plt.show()
        return importances

    def tree_predict(self, X):
        # Predict using the trees in the hidden layer
        # hidden_activations = np.tanh(np.dot(X, self.weights1) + self.bias1)
        # tree_predictions = np.column_stack(
        # [tree.predict(hidden_activations) for tree in self.trees_hidden])
        hidden_tree_outputs = np.column_stack(
            [tree.predict(X) for tree in self.trees_hidden])  # self.a1

        # Compute feature importance weights for each tree
        tree_importances = [
            tree.feature_importances_ for tree in self.trees_hidden]
        # Normalize the importances to sum to 1 for each tree
        tree_weights = np.array([importances / np.sum(importances) if np.sum(
            importances) > 0 else np.ones_like(importances) for importances in tree_importances])
        # Average the normalized importances to get the final weights for each tree
        final_weights = np.mean(tree_weights, axis=1)

        # Compute the weighted average of the tree predictions
        weighted_tree_predictions = np.average(
            hidden_tree_outputs, axis=1, weights=final_weights)
        return weighted_tree_predictions

# TREENN1: Custom MLP with a single Tree in the input layer


class TREENN1(FONN1):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(input_dim, hidden_dim, output_dim, num_trees_input=1)
# Custom MLP with 1 tree in hidden layer (TREENN2)


class TREENN2(FONN2):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__(input_dim, hidden_dim, output_dim, num_trees_hidden=1)
