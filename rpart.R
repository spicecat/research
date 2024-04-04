# LOAD NECESSARY LIBRARIES
library(MASS)
library(rpart) # for fitting decision trees
library(rpart.plot) # for plotting decision trees
library(boot)

set.seed(123)

# Create a training and test set
test_indices <- sample(seq_len(nrow(Boston)), nrow(Boston) * 0.2)
test_data <- Boston[test_indices, ]
train_data <- Boston[-test_indices, ]

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
  tree <- rpart(
    formula,
    bootstrap_sample,
    control = rpart.control(cp = 0.0001)
  )
  best <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
  pruned_tree <- prune(tree, cp = best)
  test_predictions <- predict(pruned_tree, newdata = data)
  test_predictions
}

reps <- boot(
  Boston,
  predictions,
  5,
  formula = medv ~ .
)

# print(reps)
plot(reps)

Boston_boot <- cbind(Boston, t(reps$t))
# print(Boston_boot)
