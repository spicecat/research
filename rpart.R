# LOAD NECESSARY LIBRARIES
# library(ISLR) # contains Hitters dataset
library(MASS)
library(rpart) # for fitting decision trees
library(rpart.plot) # for plotting decision trees

set.seed(1)

# REGRESSION TREE
# build the initial tree
# tree <- rpart(Salary ~ Years + HmRun, data = Hitters, control = rpart.control(cp = .0001))
tree <- rpart(medv ~ crim + zn + indus, data = Boston, control = rpart.control(cp = .0001))

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
  roundint = F, # don't round to integers in output
  digits = 5
) # display 5 decimal places in output

# new <- data.frame(Years = 7, HmRun = 4)
new <- data.frame(
  crim = c(0.04741, 0.06617),
  zn = c(0, 0),
  indus = c(11.9, 3.24),
  medv = c(11.9, 19.3)
)
predictions <- predict(pruned_tree, newdata = new)
print(predictions)