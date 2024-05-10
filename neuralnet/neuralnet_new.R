# install.packages("keras")
library(keras)

# Parameters
input_size <- 3
hidden_size1 <- 4
hidden_size2 <- 3
hidden_size3 <- 2
output_size <- 1

# Model setup
model <- keras_model_sequential() %>%
  layer_dense(units = hidden_size1, input_shape = c(input_size)) %>%
  layer_activation("relu") %>%
  layer_dense(units = hidden_size2) %>%
  layer_activation("relu") %>%
  layer_dense(units = hidden_size3) %>%
  layer_activation("relu") %>%
  layer_dense(units = output_size) %>%
  layer_activation("sigmoid")

# Compile model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

# Training data
x_train <- matrix(c(0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1), ncol = input_size)
y_train <- matrix(c(1, 1, 0, 0), ncol = 1)

# Train model
model %>% fit(x_train, y_train, epochs = 10, batch_size = 2)

# Evaluate model
x_test <- matrix(c(0, 1, 0, 1, 0, 1, 0, 0, 0), ncol = input_size)
y_test <- matrix(c(1, 0, 0), ncol = 1)

score <- model %>% evaluate(x_test, y_test, verbose = 0)

print(paste("Test loss:", score$loss))
print(paste("Test accuracy:", score$acc))
