# Load the necessary libraries
library(keras)
library(MASS)

# Load the Boston Housing dataset from the MASS library
data(Boston, package = "MASS")

# The target variable is medv
train_data <- Boston[, -which(names(Boston) %in% "medv")]
train_labels <- Boston$medv

# Normalize the data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)

# Build the model
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = dim(train_data)[2]) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1) # Linear activation function for output

# Compile the model
model %>% compile(
  optimizer = "rmsprop",
  loss = "mse",
  metrics = list("mean_absolute_error")
)

# Train the model
history <- model %>% fit(
  train_data,
  train_labels,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.2
)
