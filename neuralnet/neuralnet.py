import numpy as np
import tensorflow as tf
import random

# Set the random seeds
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# helper function- sigmoid


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return tf.nn.sigmoid(x)


# parameters
input_size = 3
hidden_size1 = 4
hidden_size2 = 3
hidden_size3 = 2
output_size = 1

# TensorFlow model setup


class MyNeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        # initialize layers
        self.dense1 = tf.keras.layers.Dense(hidden_size1, use_bias=False)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dense2 = tf.keras.layers.Dense(hidden_size2, use_bias=False)
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dense3 = tf.keras.layers.Dense(hidden_size3, use_bias=False)
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dense4 = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.batch_norm1(x)
        x = nonlin(x)
        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = nonlin(x)
        x = self.dense3(x)
        x = self.batch_norm3(x)
        x = nonlin(x)
        x = self.dense4(x)
        return nonlin(x)


# training setup
model = MyNeuralNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

# input data/output labels
X = np.array([[0, 1, 1],
              [1, 1, 0],
              [1, 0, 0],
              [1, 0, 1]], dtype=np.float32)

y = np.array([[1], [1], [0], [0]], dtype=np.float32)

# training loop (i just kept batch size small for now)
epochs = 10
batch_size = 2

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=4).batch(batch_size)

for epoch in range(epochs):
    for step, (x_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# final weights
for layer in model.layers:
    if hasattr(layer, 'kernel'):
        print(f"Weights of {layer.name}: {layer.kernel.numpy()}")

# test data and predictions
X_test = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.float32)
preds = model(X_test)
print("Test Predictions:", preds.numpy())

# testing accuracy by converting probabilities to binary

# test data
X_test = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.float32)
y_test = np.array([[1], [0], [0]], dtype=np.float32)  # True labels

preds = model(X_test)

preds_binary = (preds.numpy() > 0.5).astype(int)

accuracy = np.mean(preds_binary == y_test)
print("Test Accuracy:", accuracy)
