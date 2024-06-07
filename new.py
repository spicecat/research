import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# He initialization for weights and zero initialization for biases


def he_initialization(layer_size, next_layer_size):
    return np.random.randn(layer_size, next_layer_size) * np.sqrt(2. / layer_size), np.zeros(next_layer_size)


class SoftDecisionTreeNode:
    def __init__(self, input_dim):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        self.probs = 1 / (1 + np.exp(-z))  # Sigmoid function
        return self.probs

    def backward(self, d_out, learning_rate, X):
        dz = d_out * (self.probs * (1 - self.probs))  # Sigmoid derivative
        dw = np.dot(X.T, dz) / X.shape[0]
        db = np.mean(dz)
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db


class SimpleMLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.weights1, self.bias1 = he_initialization(input_dim, hidden_dim)
        self.weights2, self.bias2 = he_initialization(hidden_dim, output_dim)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        return self.z2  # Linear output

    def backward(self, X, y, output, learning_rate):
        output_error = output - y.reshape(-1, 1)
        d_weights2 = np.dot(self.a1.T, output_error) / X.shape[0]
        d_bias2 = np.mean(output_error, axis=0)
        hidden_error = np.dot(output_error, self.weights2.T) * (self.a1 > 0)
        d_weights1 = np.dot(X.T, hidden_error) / X.shape[0]
        d_bias1 = np.mean(hidden_error, axis=0)

        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1

# Integrate soft decision tree with simple MLP


class MLPWithSoftDecisionTree:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.tree_node = SoftDecisionTreeNode(input_dim)
        self.mlp = SimpleMLP(1, hidden_dim, output_dim)

    def forward(self, X):
        tree_output = self.tree_node.forward(X).reshape(-1, 1)
        output = self.mlp.forward(tree_output)
        return output

    def backward(self, X, y, output, learning_rate):
        output_error = output - y.reshape(-1, 1)
        self.mlp.backward(self.tree_node.probs.reshape(-1, 1),
                          y, output, learning_rate)
        mlp_hidden_error = np.dot(
            output_error, self.mlp.weights2.T) * (self.mlp.a1 > 0)
        tree_node_error = np.dot(
            mlp_hidden_error, self.mlp.weights1.T).mean(axis=1)
        self.tree_node.backward(tree_node_error, learning_rate, X)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')


if __name__ == "__main__":
    # Load the dataset
    housing = fetch_california_housing()
    X = housing.data  # type: ignore
    y = housing.target  # type: ignore

    # Standardize the dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    # Training the model
    input_dim = X_train.shape[1]
    hidden_dim = 10
    output_dim = 1
    epochs = 2000
    learning_rate = 0.001

    model = MLPWithSoftDecisionTree(input_dim, hidden_dim, output_dim)
    model.train(X_train, y_train, epochs, learning_rate)

    # Making predictions
    predictions = model.forward(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(set(predictions.flatten()))
    print(r2, mae)
