import numpy as np
from sklearn.tree import DecisionTreeRegressor


class FONN2:
    def __init__(
        self, input_dim, hidden_dim, output_dim, X_train, y_train, num_trees=3
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_trees = num_trees

        # Initialize weights and biases for the input to hidden layer
        self.weights_hidden = np.random.randn(input_dim, hidden_dim) * np.sqrt(
            2.0 / input_dim
        )
        self.bias_hidden = np.zeros(hidden_dim)

        # Initialize and train multiple decision trees for the hidden layer
        self.trees_hidden = [
            DecisionTreeRegressor(max_depth=5, random_state=i).fit(X_train, y_train)
            for i in range(num_trees)
        ]

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.randn(
            hidden_dim + num_trees, output_dim
        ) * np.sqrt(2.0 / (hidden_dim + num_trees))
        self.bias_output = np.zeros(output_dim)

    def forward(self, X):
        # Compute hidden layer activations
        z_hidden = np.dot(X, self.weights_hidden) + self.bias_hidden
        a_hidden = np.tanh(z_hidden)  # Activation function

        # Predict with the trees and concatenate their outputs with hidden layer activations
        tree_outputs = [tree.predict(X).reshape(-1, 1) for tree in self.trees_hidden]
        self.combined_hidden = np.hstack([a_hidden] + tree_outputs)

        # Compute output layer activations
        z_output = np.dot(self.combined_hidden, self.weights_output) + self.bias_output
        return z_output  # Linear activation

    def train(self, X, y, epochs, learning_rate, gradient_clip=1.0):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)

            # Backward pass
            output_error = output - y.reshape(-1, 1)
            d_weights_output = np.dot(self.combined_hidden.T, output_error) / X.shape[0]
            d_bias_output = np.mean(output_error, axis=0)
            hidden_error = np.dot(output_error, self.weights_output.T)[
                :, : -self.num_trees
            ] * (1 - np.tanh(np.dot(X, self.weights_hidden) + self.bias_hidden) ** 2)
            d_weights_hidden = np.dot(X.T, hidden_error) / X.shape[0]
            d_bias_hidden = np.mean(hidden_error, axis=0)

            # Gradient Clipping
            d_weights_output = np.clip(d_weights_output, -gradient_clip, gradient_clip)
            d_bias_output = np.clip(d_bias_output, -gradient_clip, gradient_clip)
            d_weights_hidden = np.clip(d_weights_hidden, -gradient_clip, gradient_clip)
            d_bias_hidden = np.clip(d_bias_hidden, -gradient_clip, gradient_clip)

            # Update weights and biases
            self.weights_output -= learning_rate * d_weights_output
            self.bias_output -= learning_rate * d_bias_output
            self.weights_hidden -= learning_rate * d_weights_hidden
            self.bias_hidden -= learning_rate * d_bias_hidden

            # if epoch % 100 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss}")
