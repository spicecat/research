import numpy as np
from sklearn.tree import DecisionTreeRegressor

class TREENN1:
    def __init__(self, input_dim, hidden_dim, output_dim, X_train, y_train):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize and train the decision tree for the input layer
        self.tree_input = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X_train, y_train)

        # Initialize weights and biases
        self.weights_hidden = np.random.randn(input_dim + 1, hidden_dim) * np.sqrt(2. / (input_dim + 1))  # He initialization
        self.bias_hidden = np.zeros(hidden_dim)
        self.weights_output = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / hidden_dim)
        self.bias_output = np.zeros(output_dim)

    def forward(self, X):
        # Predict with the tree at the input layer
        tree_output = self.tree_input.predict(X).reshape(-1, 1)

        # Concatenate tree output with input features
        self.combined_input = np.hstack((X, tree_output))

        # Compute hidden layer activations
        z_hidden = np.dot(self.combined_input, self.weights_hidden) + self.bias_hidden
        self.a_hidden = np.tanh(z_hidden)  # Store the activation for backpropagation

        # Compute output layer activations
        z_output = np.dot(self.a_hidden, self.weights_output) + self.bias_output
        return z_output  # Linear activation

    def train(self, X, y, epochs, learning_rate, gradient_clip=1.0):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)

            # Backward pass
            output_error = output - y.reshape(-1, 1)
            d_weights_output = np.dot(self.a_hidden.T, output_error) / X.shape[0]
            d_bias_output = np.mean(output_error, axis=0)
            hidden_error = np.dot(output_error, self.weights_output.T) * (1 - self.a_hidden ** 2)
            d_weights_hidden = np.dot(self.combined_input.T, hidden_error) / X.shape[0]
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

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
