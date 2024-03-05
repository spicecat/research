import numpy as np


def nonlin(x, deriv=False):
    if deriv == True:
        y = 1 / (1 + np.exp(-x))
        return y * (1 - y)
    return 1 / (1 + np.exp(-x))


# input data/output labels
X = np.array([[0, 1, 1],
              [1, 1, 0],
              [1, 0, 0],
              [1, 0, 1]])

y = np.array([[1], [1], [0], [0]])

# setting random seed for reproducing
rng = np.random.default_rng(1)

# initialize weights
w0 = 2 * rng.random((3, 4)) - 1
w1 = 2 * rng.random((4, 1)) - 1

# hyperparameters
learning_rate = 0.01
epochs = 10
batch_size = 100
iterations_per_epoch = 10000

total_iterations = 0

# training loop
for epoch in range(epochs):
    # random batch selection
    indices = rng.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for batch_start in range(0, len(X), batch_size):
        # Select a random batch
        batch_end = batch_start + batch_size
        X_batch = X_shuffled[batch_start:batch_end]
        y_batch = y_shuffled[batch_start:batch_end]

        # forward prop
        a0 = X_batch
        a1 = nonlin(np.dot(a0, w0))
        a2 = nonlin(np.dot(a1, w1))

        # backprop
        a2_error = (1/2) * np.square(y_batch - a2)
        a2_delta = a2_error * nonlin(a2, deriv=True)
        a1_error = a2_delta.dot(w1.T)
        a1_delta = a1_error * nonlin(a1, deriv=True)

        # update weights
        w1 += a1.T.dot(a2_delta) * learning_rate
        w0 += a0.T.dot(a1_delta) * learning_rate

        total_iterations += 1
        if total_iterations == iterations_per_epoch:
            break  # exit the loop after reaching the desired iterations

    # error val after each epoch
    print(f"Epoch {epoch + 1}, Error: {np.mean(np.abs(a2_error))}")

# final weights
print("Final weights:")
print("w0:", w0)
print("w1:", w1)


def predict(X):
    a1 = nonlin(np.dot(X, w0))
    a2 = nonlin(np.dot(a1, w1))
    return a2


# test data
X_test = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])
y_test = np.array([[1], [0], [0]])

# predictions
preds = predict(X_test)

# evaluate accuracy
acc = np.mean(preds == y_test)
print("Test Accuracy:", acc)
