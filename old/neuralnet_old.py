import keras
import numpy as np
import tensorflow as tf
import random


def nonlin(x, deriv=False):
    # helper function- sigmoid
    if deriv:
        return x * (1 - x)
    return tf.nn.sigmoid(x)

# parameters
input_size = 3
hidden_size1 = 4
hidden_size2 = 3
hidden_size3 = 2
output_size = 1

class MyNeuralNetwork(keras.Model):
    # TensorFlow model setup
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()
        # initialize layers
        self.dense1 = keras.layers.Dense(hidden_size1, use_bias=False)
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(hidden_size2, use_bias=False)
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.dense3 = keras.layers.Dense(hidden_size3, use_bias=False)
        self.batch_norm3 = keras.layers.BatchNormalization()
        self.dense4 = keras.layers.Dense(output_size)

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


if __name__ == "__main__":
    # Set the random seeds
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # training setup
    model = MyNeuralNetwork()
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = keras.losses.MeanSquaredError()

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
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
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
