import numpy as np
from sklearn.tree import DecisionTreeRegressor


class FONN1:
    def __init__(self, input_dim, hidden_dim, output_dim, num_trees=3, learning_rate_init=0.01, max_iter=1000, alpha=0.5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_trees = num_trees
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.alpha = alpha  # Scaling factor for tree outputs
        self.loss_curve_ = []

        # Initialize and train multiple decision trees for the input layer
        self.trees_input = [
            DecisionTreeRegressor(
                max_depth=5, random_state=i)
            for i in range(num_trees)
        ]

        # Initialize weights and biases for the hidden layer
        self.weights_hidden = np.random.randn(
            input_dim + num_trees, hidden_dim) * np.sqrt(2. / (input_dim + num_trees))
        self.bias_hidden = np.zeros(hidden_dim)

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.randn(
            hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.bias_output = np.zeros(output_dim)

    def _forward(self, X):
        # Predict with each tree and scale outputs
        tree_outputs = [tree.predict(X).reshape(-1, 1)
                        for tree in self.trees_input]
        tree_outputs_scaled = [(t - t.mean()) / (t.std() + 1e-8)
                               for t in tree_outputs]

        # Combine input features with scaled tree outputs
        self.combined_input = np.hstack(
            [X] + [self.alpha * t for t in tree_outputs_scaled])

        # Compute hidden layer activations
        self.z_hidden = np.dot(self.combined_input,
                               self.weights_hidden) + self.bias_hidden
        self.a_hidden = np.tanh(self.z_hidden)  # Activation function

        # Compute output layer activations
        self.z_output = np.dot(
            self.a_hidden, self.weights_output) + self.bias_output
        return self.z_output  # Linear activation

    # def _backward(self, y, activations):
    #     # Backward pass
    #     output_error = activations - y.reshape(-1, 1)
    #     d_weights_output = np.dot(
    #         self.a_hidden.T, output_error) / y.shape[0]
    #     d_bias_output = np.mean(output_error, axis=0)
    #     hidden_error = np.dot(
    #         output_error, self.weights_output.T) * (1 - self.a_hidden ** 2)
    #     d_weights_hidden = np.dot(
    #         self.combined_input.T, hidden_error) / y.shape[0]
    #     d_bias_hidden = np.mean(hidden_error, axis=0)

    #     # Gradient Clipping
    #     max_grad_norm = 1.0
    #     d_weights_output = np.clip(
    #         d_weights_output, -max_grad_norm, max_grad_norm)
    #     d_bias_output = np.clip(
    #         d_bias_output, -max_grad_norm, max_grad_norm)
    #     d_weights_hidden = np.clip(
    #         d_weights_hidden, -max_grad_norm, max_grad_norm)
    #     d_bias_hidden = np.clip(
    #         d_bias_hidden, -max_grad_norm, max_grad_norm)

    #     # Update weights and biases
    #     self.weights_output -= self.learning_rate * d_weights_output
    #     self.bias_output -= self.learning_rate * d_bias_output
    #     self.weights_hidden -= self.learning_rate * d_weights_hidden
    #     self.bias_hidden -= self.learning_rate * d_bias_hidden

    def fit(self, X, y):
        for tree in self.trees_input:
            tree.fit(X, y)

        for epoch in range(self.max_iter):
            # Forward pass
            output = self._forward(X)

            # Compute loss
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)

            # Backward pass
            output_error = output - y.reshape(-1, 1)
            d_weights_output = np.dot(
                self.a_hidden.T, output_error) / X.shape[0]
            d_bias_output = np.mean(output_error, axis=0)
            hidden_error = np.dot(
                output_error, self.weights_output.T) * (1 - self.a_hidden ** 2)
            d_weights_hidden = np.dot(
                self.combined_input.T, hidden_error) / X.shape[0]
            d_bias_hidden = np.mean(hidden_error, axis=0)

            # Gradient Clipping
            max_grad_norm = 1.0
            d_weights_output = np.clip(
                d_weights_output, -max_grad_norm, max_grad_norm)
            d_bias_output = np.clip(
                d_bias_output, -max_grad_norm, max_grad_norm)
            d_weights_hidden = np.clip(
                d_weights_hidden, -max_grad_norm, max_grad_norm)
            d_bias_hidden = np.clip(
                d_bias_hidden, -max_grad_norm, max_grad_norm)

            # Update weights and biases
            self.weights_output -= self.learning_rate * d_weights_output
            self.bias_output -= self.learning_rate * d_bias_output
            self.weights_hidden -= self.learning_rate * d_weights_hidden
            self.bias_hidden -= self.learning_rate * d_bias_hidden

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self._forward(X)


class FONN2:
    def __init__(self, input_dim, hidden_dim, output_dim, num_trees=3, learning_rate_init=0.01, max_iter=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_trees = num_trees
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.loss_curve_ = []

        # Initialize weights and biases for the input to hidden layer
        self.weights_hidden = np.random.randn(
            input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.bias_hidden = np.zeros(hidden_dim)

        # Initialize and train multiple decision trees for the hidden layer
        self.trees_hidden = [
            DecisionTreeRegressor(
                max_depth=5, random_state=i)
            for i in range(num_trees)
        ]

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.randn(
            hidden_dim + num_trees, output_dim) * np.sqrt(2. / (hidden_dim + num_trees))
        self.bias_output = np.zeros(output_dim)

    def _forward(self, X):
        # Compute hidden layer activations
        z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        a_hidden = np.tanh(z_hidden)  # Activation function

        # Predict with the trees and concatenate their outputs with hidden layer activations
        tree_outputs = [tree.predict(X).reshape(-1, 1)
                        for tree in self.trees_hidden]
        self.combined_hidden = np.hstack([a_hidden] + tree_outputs)

        # Compute output layer activations
        z_output = np.dot(self.combined_hidden,
                          self.weights_output) + self.bias_output
        return z_output  # Linear activation

    def fit(self, X, y):
        for tree in self.trees_hidden:
            tree.fit(X, y)

        for epoch in range(self.max_iter):
            # Forward pass
            output = self._forward(X)

            # Compute loss
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)

            # Backward pass
            output_error = output - y.reshape(-1, 1)
            d_weights_output = np.dot(
                self.combined_hidden.T, output_error) / X.shape[0]
            d_bias_output = np.mean(output_error, axis=0)
            hidden_error = np.dot(output_error, self.weights_output.T)[
                :, :-self.num_trees] * (1 - np.tanh(np.dot(X, self.weights_hidden) + self.bias_hidden) ** 2)
            d_weights_hidden = np.dot(X.T, hidden_error) / X.shape[0]
            d_bias_hidden = np.mean(hidden_error, axis=0)

            # Gradient Clipping
            max_grad_norm = 1.0
            d_weights_output = np.clip(
                d_weights_output, -max_grad_norm, max_grad_norm)
            d_bias_output = np.clip(
                d_bias_output, -max_grad_norm, max_grad_norm)
            d_weights_hidden = np.clip(
                d_weights_hidden, -max_grad_norm, max_grad_norm)
            d_bias_hidden = np.clip(
                d_bias_hidden, -max_grad_norm, max_grad_norm)

            # Update weights and biases
            self.weights_output -= self.learning_rate * d_weights_output
            self.bias_output -= self.learning_rate * d_bias_output
            self.weights_hidden -= self.learning_rate * d_weights_hidden
            self.bias_hidden -= self.learning_rate * d_bias_hidden

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self._forward(X)


class FONN3:
    def __init__(self, input_dim, hidden_dim, output_dim, num_trees=3, learning_rate_init=0.01, max_iter=1000, alpha=1.):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_trees = num_trees
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.alpha = alpha  # Scaling factor for tree contributions
        self.loss_curve_ = []

        # Initialize weights and biases for the input to hidden layer
        self.weights_hidden = np.random.randn(
            input_dim, hidden_dim) * np.sqrt(2. / input_dim)
        self.bias_hidden = np.zeros(hidden_dim)

        # Initialize weights and biases for the hidden to pre-output layer
        self.weights_output = np.random.randn(
            hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.bias_output = np.zeros(output_dim)

        # Initialize and train multiple decision trees for the output layer
        self.trees_output = [
            DecisionTreeRegressor(
                max_depth=5, random_state=i)
            for i in range(num_trees)
        ]

    def _forward(self, X):
        # Compute hidden layer activations
        z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.a_hidden = np.tanh(z_hidden)  # Activation function

        # Compute pre-output layer activations
        self.z_output = np.dot(
            self.a_hidden, self.weights_output) + self.bias_output

        # Predict with the trees and normalize their outputs
        tree_outputs = [tree.predict(X).reshape(-1, 1)
                        for tree in self.trees_output]
        tree_outputs_scaled = [(t - t.mean()) / (t.std() + 1e-8)
                               for t in tree_outputs]

        # Combine the neural network and tree outputs
        self.combined_output = self.z_output + \
            self.alpha * sum(tree_outputs_scaled)
        return self.combined_output

    def fit(self, X, y):
        for tree in self.trees_output:
            tree.fit(X, y)

        for epoch in range(self.max_iter):
            # Forward pass
            output = self._forward(X)

            # Compute loss
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)

            # Backward pass
            output_error = output - y.reshape(-1, 1)
            d_weights_output = np.dot(
                self.a_hidden.T, output_error) / X.shape[0]
            d_bias_output = np.mean(output_error, axis=0)
            hidden_error = np.dot(
                output_error, self.weights_output.T) * (1 - self.a_hidden ** 2)
            d_weights_hidden = np.dot(X.T, hidden_error) / X.shape[0]
            d_bias_hidden = np.mean(hidden_error, axis=0)

            # Gradient Clipping
            max_grad_norm = 1.0
            d_weights_output = np.clip(
                d_weights_output, -max_grad_norm, max_grad_norm)
            d_bias_output = np.clip(
                d_bias_output, -max_grad_norm, max_grad_norm)
            d_weights_hidden = np.clip(
                d_weights_hidden, -max_grad_norm, max_grad_norm)
            d_bias_hidden = np.clip(
                d_bias_hidden, -max_grad_norm, max_grad_norm)

            # Update weights and biases
            self.weights_output -= self.learning_rate * d_weights_output
            self.bias_output -= self.learning_rate * d_bias_output
            self.weights_hidden -= self.learning_rate * d_weights_hidden
            self.bias_hidden -= self.learning_rate * d_bias_hidden

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self._forward(X)


class TREENN1:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate_init=0.01, max_iter=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.loss_curve_ = []

        # Initialize and train the decision tree for the input layer
        self.tree_input = DecisionTreeRegressor(
            max_depth=5, random_state=0)

        # Initialize weights and biases
        self.weights_hidden = np.random.randn(
            input_dim + 1, hidden_dim) * np.sqrt(2. / (input_dim + 1))  # He initialization
        self.bias_hidden = np.zeros(hidden_dim)
        self.weights_output = np.random.randn(
            hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.bias_output = np.zeros(output_dim)

    def _forward(self, X):
        # Predict with the tree at the input layer
        tree_output = self.tree_input.predict(X).reshape(-1, 1)

        # Concatenate tree output with input features
        self.combined_input = np.hstack((X, tree_output))

        # Compute hidden layer activations
        z_hidden = np.dot(self.combined_input,
                          self.weights_hidden) + self.bias_hidden
        # Store the activation for backpropagation
        self.a_hidden = np.tanh(z_hidden)

        # Compute output layer activations
        z_output = np.dot(
            self.a_hidden, self.weights_output) + self.bias_output
        return z_output  # Linear activation

    def fit(self, X, y):
        self.tree_input.fit(X, y)

        for epoch in range(self.max_iter):
            # Forward pass
            output = self._forward(X)

            # Compute loss
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)

            # Backward pass
            output_error = output - y.reshape(-1, 1)
            d_weights_output = np.dot(
                self.a_hidden.T, output_error) / X.shape[0]
            d_bias_output = np.mean(output_error, axis=0)
            hidden_error = np.dot(
                output_error, self.weights_output.T) * (1 - self.a_hidden ** 2)
            d_weights_hidden = np.dot(
                self.combined_input.T, hidden_error) / X.shape[0]
            d_bias_hidden = np.mean(hidden_error, axis=0)

            # Gradient Clipping
            max_grad_norm = 1.0
            d_weights_output = np.clip(
                d_weights_output, -max_grad_norm, max_grad_norm)
            d_bias_output = np.clip(
                d_bias_output, -max_grad_norm, max_grad_norm)
            d_weights_hidden = np.clip(
                d_weights_hidden, -max_grad_norm, max_grad_norm)
            d_bias_hidden = np.clip(
                d_bias_hidden, -max_grad_norm, max_grad_norm)

            # Update weights and biases
            self.weights_output -= self.learning_rate * d_weights_output
            self.bias_output -= self.learning_rate * d_bias_output
            self.weights_hidden -= self.learning_rate * d_weights_hidden
            self.bias_hidden -= self.learning_rate * d_bias_hidden

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self._forward(X)


class TREENN2:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate_init=0.01, max_iter=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.loss_curve_ = []

        # Initialize and train the decision tree for the hidden layer
        self.tree_hidden = DecisionTreeRegressor(
            max_depth=5, random_state=0)

        # Initialize weights and biases
        self.weights_hidden = np.random.randn(
            input_dim, hidden_dim) * np.sqrt(2. / input_dim)  # He initialization
        self.bias_hidden = np.zeros(hidden_dim)
        self.weights_output = np.random.randn(
            hidden_dim + 1, output_dim) * np.sqrt(2. / hidden_dim)
        self.bias_output = np.zeros(output_dim)

    def _forward(self, X):
        # Compute hidden layer activations
        z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        a_hidden = np.tanh(z_hidden)  # Activation function

        # Predict with the tree and concatenate its output with hidden layer activations
        hidden_tree_output = self.tree_hidden.predict(X).reshape(-1, 1)
        self.combined_hidden = np.hstack((a_hidden, hidden_tree_output))

        # Compute output layer activations
        z_output = np.dot(self.combined_hidden,
                          self.weights_output) + self.bias_output
        return z_output  # Linear activation

    def fit(self, X, y):
        self.tree_hidden.fit(X, y)

        for epoch in range(self.max_iter):
            # Forward pass
            output = self._forward(X)

            # Compute loss
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)

            # Backward pass
            output_error = output - y.reshape(-1, 1)
            d_weights_output = np.dot(
                self.combined_hidden.T, output_error) / X.shape[0]
            d_bias_output = np.mean(output_error, axis=0)
            hidden_error = np.dot(output_error, self.weights_output.T)[
                :, :-1] * (1 - np.tanh(np.dot(X, self.weights_hidden) + self.bias_hidden) ** 2)
            d_weights_hidden = np.dot(X.T, hidden_error) / X.shape[0]
            d_bias_hidden = np.mean(hidden_error, axis=0)

            # Gradient Clipping
            max_grad_norm = 1.0
            d_weights_output = np.clip(
                d_weights_output, -max_grad_norm, max_grad_norm)
            d_bias_output = np.clip(
                d_bias_output, -max_grad_norm, max_grad_norm)
            d_weights_hidden = np.clip(
                d_weights_hidden, -max_grad_norm, max_grad_norm)
            d_bias_hidden = np.clip(
                d_bias_hidden, -max_grad_norm, max_grad_norm)

            # Update weights and biases
            self.weights_output -= self.learning_rate * d_weights_output
            self.bias_output -= self.learning_rate * d_bias_output
            self.weights_hidden -= self.learning_rate * d_weights_hidden
            self.bias_hidden -= self.learning_rate * d_bias_hidden

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self._forward(X)


class TREENN3:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate_init=0.01, max_iter=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate_init
        self.max_iter = max_iter
        self.loss_curve_ = []

        # Initialize weights and biases for the input to hidden layer
        self.weights_hidden = np.random.randn(
            input_dim, hidden_dim) * np.sqrt(2. / input_dim)  # He initialization
        self.bias_hidden = np.zeros(hidden_dim)

        # Initialize weights and biases for the hidden to pre-output layer
        self.weights_output = np.random.randn(
            hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.bias_output = np.zeros(output_dim)

        # Initialize and train the decision tree for the output layer
        self.tree_output = DecisionTreeRegressor(
            max_depth=5, random_state=0)

    def _forward(self, X):
        # Compute hidden layer activations
        z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.a_hidden = np.tanh(z_hidden)  # Activation function

        # Compute pre-output layer activations
        self.z_output = np.dot(
            self.a_hidden, self.weights_output) + self.bias_output

        # Predict with the tree and combine its output with pre-output activations
        tree_output = self.tree_output.predict(X).reshape(-1, 1)
        # Combining neural and tree outputs
        self.combined_output = self.z_output + tree_output

        return self.combined_output

    def fit(self, X, y):
        self.tree_output.fit(X, y)

        for epoch in range(self.max_iter):
            # Forward pass
            output = self._forward(X)

            # Compute loss
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)

            # Backward pass
            output_error = output - y.reshape(-1, 1)
            d_weights_output = np.dot(
                self.a_hidden.T, output_error) / X.shape[0]
            d_bias_output = np.mean(output_error, axis=0)
            hidden_error = np.dot(
                output_error, self.weights_output.T) * (1 - self.a_hidden ** 2)
            d_weights_hidden = np.dot(X.T, hidden_error) / X.shape[0]
            d_bias_hidden = np.mean(hidden_error, axis=0)

           # Gradient Clipping
            max_grad_norm = 1.0
            d_weights_output = np.clip(
                d_weights_output, -max_grad_norm, max_grad_norm)
            d_bias_output = np.clip(
                d_bias_output, -max_grad_norm, max_grad_norm)
            d_weights_hidden = np.clip(
                d_weights_hidden, -max_grad_norm, max_grad_norm)
            d_bias_hidden = np.clip(
                d_bias_hidden, -max_grad_norm, max_grad_norm)

            # Update weights and biases
            self.weights_output -= self.learning_rate * d_weights_output
            self.bias_output -= self.learning_rate * d_bias_output
            self.weights_hidden -= self.learning_rate * d_weights_hidden
            self.bias_hidden -= self.learning_rate * d_bias_hidden

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        return self._forward(X)
