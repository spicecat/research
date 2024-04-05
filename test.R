# LOAD NECESSARY LIBRARIES
packages <- c(
  "MASS", "boot", "dplyr", "shapr", "rpart", "rpart.plot", "xgboost"
)
installed_packages <- packages %in% rownames(installed.packages())
install.packages(packages[!installed_packages])
lapply(packages, library, character.only = TRUE)

set.seed(123)

# included_columns <- c("lstat", "rm", "dis", "indus", "medv")
included_columns <- c("indus", "chas", "rm", "age", "tax", "medv")
data <- Boston[, included_columns]
# data <- Boston
data$chas <- as.factor(data$chas)
data <- data %>% mutate_if(is.factor, as.numeric)

predictions <- function(formula, data, indices) {
  bootstrap_sample <- data[indices, ]
  # build the initial tree
  tree <- rpart( # nolint: object_usage_linter.
    formula,
    data = bootstrap_sample,
    control = rpart.control(cp = 0.0001) # nolint: object_usage_linter.
  )
  # identify best cp value to use
  best <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
  # produce a pruned tree based on the best cp value
  pruned_tree <- prune(tree, cp = best) # nolint: object_usage_linter.
  # Generate predictions on the test subset
  test_predictions <- predict(pruned_tree, newdata = data)
  test_predictions
}

reps <- boot(
  data,
  statistic = predictions,
  R = 5,
  formula = medv ~ .
)

# print(reps)
# plot(reps)

data_boot <- cbind(t(reps$t), data)
# Rename the columns
colnames(data_boot)[
  seq_len(nrow(reps$t))
] <- paste0("p", seq_len(nrow(reps$t)))
# print(data_boot)
data_boot

plot_shap <- function(data, y_vars, x_vars, test_size = 0.2) {
  # Create a training and test set
  test_indices <- sample(seq_len(nrow(data_boot)), nrow(data) * test_size)
  x_train <- as.matrix(data_boot[-test_indices, x_vars])
  y_train <- data_boot[-test_indices, y_vars]
  x_test <- as.matrix(data_boot[test_indices, x_vars])

  # Convert the data to xgb.DMatrix format
  dtrain <- xgb.DMatrix(data = x_train, label = y_train) # nolint: object_usage_linter.

  # https://xgboost.readthedocs.io/en/stable/parameter.html
  # Set up the parameters for the xgboost model
  params <- list(
    objective = "reg:squarederror",
    eta = 0.3
  )

  # Train the xgboost model
  model <- xgb.train( # nolint: object_usage_linter.
    params = params,
    data = dtrain,
    nrounds = 20,
    watchlist = list(train = dtrain)
  )
  # model <- xgboost(
  #   data = x_train,
  #   label = y_train,
  #   nround = 20,
  #   verbose = FALSE
  # )

  # Prepare the data for explanation
  explainer <- shapr(
    x_train,
    model,
    n_combinations = 10000
  )

  # Specifying the phi_0, i.e. the expected prediction without any features
  p <- mean(y_train)

  # Computing the actual Shapley values with kernelSHAP accounting for feature dependence using
  # the empirical (conditional) distribution approach with bandwidth parameter sigma = 0.1 (default)
  explanation <- explain(
    x_test,
    approach = "empirical",
    explainer = explainer,
    prediction_zero = p
  )
  # Printing the Shapley values for the test data.
  print(explanation$dt)

  # Plot the resulting explanations for observations
  timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
  png(paste0("output/shap_", timestamp, ".png"))
  plot(explanation, plot_phi0 = FALSE, index_x_test = head(test_indices))
  dev.off()
}

plot_shap(data_boot, "medv", setdiff(names(data_boot), "medv"))
plot_shap(data, "medv", setdiff(names(data), "medv"))
