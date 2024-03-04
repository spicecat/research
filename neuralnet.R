library(matrixStats)

nonlin <- function(x, deriv = FALSE) {
    if (deriv == TRUE) {
        y <- 1 / (1 + exp(-x))
        return(y * (1 - y))
    }
    return(1 / (1 + exp(-x)))
}

# input data/output labels
X <- matrix(c(
    0, 1, 1,
    1, 1, 0,
    1, 0, 0,
    1, 0, 1
), nrow = 4, byrow = TRUE)

y <- matrix(c(1, 1, 0, 0), nrow = 4)

# setting random seed for reproducing
set.seed(1)

# initialize weights
w0 <- 2 * matrix(runif(12), nrow = 3) - 1
w1 <- 2 * matrix(runif(4), nrow = 4) - 1
# hyperparameters
learning_rate <- 0.01
epochs <- 10
batch_size <- 100
iterations_per_epoch <- 10000

total_iterations <- 0

# training loop
for (epoch in 1:epochs) {
    # random batch selection
    indices <- sample(nrow(X))
    X_shuffled <- X[indices, ]
    y_shuffled <- y[indices, ]

    for (batch_start in seq(1, nrow(X), by = batch_size)) {
        # Select a random batch
        batch_end <- min(nrow(X), batch_start + batch_size - 1)

        X_batch <- X_shuffled[batch_start:batch_end, ]
        y_batch <- y_shuffled[batch_start:batch_end]

        # forward prop
        a0 <- X_batch
        a1 <- nonlin(a0 %*% w0)
        a2 <- nonlin(a1 %*% w1)

        # backprop
        a2_error <- (1 / 2) * (y_batch - a2)^2
        a2_delta <- a2_error * nonlin(a2, deriv = TRUE)
        a1_error <- a2_delta %*% t(w1)
        a1_delta <- a1_error * nonlin(a1, deriv = TRUE)

        # update weights
        w1 <- w1 + t(a1) %*% a2_delta * learning_rate
        w0 <- w0 + t(a0) %*% a1_delta * learning_rate

        total_iterations <- total_iterations + 1
        if (total_iterations == iterations_per_epoch) {
            break # exit the loop after reaching the desired iterations
        }
    }

    # error val after each epoch
    print(paste("Epoch", epoch, ", Error:", mean(abs(a2_error))))
}

# final weights
print("Final weights:")
print("w0:")
print(w0)
print("w1:")
print(w1)
