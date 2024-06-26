# LOAD NECESSARY LIBRARIES
packages <- c("MASS", "boot", "dplyr", "rpart", "rpart.plot", "neuralnet")
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}
lapply(packages, library, character.only = TRUE)


set.seed(123)

data <- Boston
data$chas <- as.factor(data$chas)
data <- data %>% mutate_if(is.factor, as.numeric)

# Create a training and test set
test_indices <- sample(seq_len(nrow(data)), nrow(data) * 0.2)
test_data <- data[test_indices, ]
train_data <- data[-test_indices, ]

# REGRESSION TREE
# build the initial tree
tree <- rpart(medv ~ ., data = train_data, control = rpart.control(cp = 0.0001))

# view results of model
# printcp(tree)

# identify best cp value to use
best <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]

# produce a pruned tree based on the best cp value
pruned_tree <- prune(tree, cp = best)

# plot the pruned tree
prp(pruned_tree,
  faclen = 0, # use full names for factor labels
  extra = 1, # display number of obs. for each terminal node
  roundint = FALSE, # don't round to integers in output
  digits = 5
) # display 5 decimal places in output

# Generate predictions on the test subset
test_predictions <- predict(pruned_tree, newdata = test_data)

# Print the predictions
# print(test_predictions)

predictions <- function(formula, data, indices) {
  bootstrap_sample <- data[indices, ]
  tree <- rpart( # nolint: object_usage_linter.
    formula,
    data = bootstrap_sample,
    control = rpart.control(cp = 0.0001) # nolint: object_usage_linter.
  )
  best <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
  pruned_tree <- prune(tree, cp = best) # nolint: object_usage_linter.
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

data_boot <- cbind(data, t(reps$t))
# Rename the columns
colnames(data_boot)[
  (ncol(data) + 1):ncol(data_boot)
] <- paste0("p", seq_len(nrow(reps$t)))
# print(data_boot)

n <- neuralnet(
  medv ~ .,
  data = data_boot,
  hidden = c(12, 7),
  linear.output = FALSE,
  lifesign = "full",
  rep = 1
)

plot(
  n,
  col.hidden = "darkgreen",
  col.hidden.synapse = "darkgreen",
  show.weights = FALSE,
  information = FALSE,
  fill = "lightblue"
)
