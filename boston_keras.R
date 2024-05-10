# LOAD NECESSARY LIBRARIES
packages <- c("boot", "conflicted", "neuralnet", "rpart", "rpart.plot", "tidyverse")
installed_packages <- packages %in% rownames(installed.packages())
install.packages(packages[!installed_packages])
lapply(packages, library, character.only = TRUE)

# LOAD BOSTON HOUSING DATA
data(Boston, package = "MASS")
print(summary(Boston))

set.seed(245)

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

tree_neuralnet <- function(
    formula,
    data,
    train_size = 0.8,
    trees = 5,
    hidden = c(10)) {
  data <- data %>% mutate_if(is.character, as.factor)

  data_rows <- floor(train_size * nrow(data))
  train_indices <- sample(c(seq_len(nrow(data))), data_rows)
  train_data <- Boston[train_indices, ]
  test_data <- Boston[-train_indices, ]

  reps <- boot(
    data,
    statistic = predictions,
    R = 5L,
    formula = medv ~ .
  )

  # TRAIN NEURAL NETWORK
  model <- neuralnet(
    formula,
    data = train_data,
    hidden = hidden,
    threshold = 0.5,
    stepmax = 1e6
  )

  # # PLOT MODEL
  # plot(model, rep = "best")

  # PREDICT MEDV
  pred <- predict(model, test_data)

  # CALCULATE ACCURACY
  mse <- mean((test_data$medv - pred)^2)
  print(paste("Mean Squared Error: ", mse))
}

tree_neuralnet(medv ~ ., Boston)
