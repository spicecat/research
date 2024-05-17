# LOAD NECESSARY LIBRARIES
packages <- c(
  "MASS", "SHAPforxgboost", "boot", "dplyr", "rpart", "rpart.plot", "xgboost"
)
installed_packages <- packages %in% rownames(installed.packages())
install.packages(packages[!installed_packages])
lapply(packages, library, character.only = TRUE)

set.seed(123)

included_columns <- c("indus", "chas", "rm", "age", "tax", "medv")
data <- Boston[, included_columns]
# data <- Boston
data$chas <- as.factor(data$chas)
data <- data %>% mutate_if(is.factor, as.numeric)

replicate_i <- 0
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
  print(tree$cptable)
  # produce a pruned tree based on the best cp value
  pruned_tree <- prune(tree, cp = best) # nolint: object_usage_linter.

  timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
  pdf(paste0("output/tree", replicate_i, "_", timestamp, ".pdf"))
  # plot the pruned tree
  prp(pruned_tree,
    faclen = 0, # use full names for factor labels
    extra = 1, # display number of obs. for each terminal node
    roundint = FALSE, # don't round to integers in output
    digits = 5
  ) # display 5 decimal places in output
  dev.off()
  replicate_i <<- replicate_i + 1

  # Generate predictions on the test subset
  test_predictions <- predict(pruned_tree, newdata = data)
  test_predictions
}

reps <- boot(
  data,
  statistic = predictions,
  R = 5L,
  formula = medv ~ .
)

# print(reps)
# plot(reps)

boot_only <- t(reps$t)
# Rename the columns
colnames(boot_only) <- paste0("p", seq_len(ncol(boot_only)))
boot_only <- cbind(data$medv, boot_only)
colnames(boot_only)[1] <- "medv"

data_boot <- cbind(boot_only, data)
# print(data_boot)

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

  shap <- shap.prep(model, X_train = x_train)
  shap.plot.summary(shap)
  shap
}

shap_boot_only <- plot_shap(
  boot_only,
  "medv",
  setdiff(colnames(boot_only), "medv")
)
shap_data_boot <- plot_shap(
  data_boot,
  "medv",
  setdiff(colnames(data_boot), "medv")
)
shap_data <- plot_shap(
  data,
  "medv",
  setdiff(colnames(data), "medv")
)

timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
pdf(paste0("output/boot_only_", timestamp, ".pdf"))
shap.plot.summary(shap_boot_only)
dev.off()

timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
pdf(paste0("output/shap_boot_", timestamp, ".pdf"))
shap.plot.summary(shap_data_boot)
dev.off()

timestamp <- format(Sys.time(), "%Y%m%d%H%M%S")
pdf(paste0("output/shap", timestamp, ".pdf"))
shap.plot.summary(shap_data)
dev.off()
