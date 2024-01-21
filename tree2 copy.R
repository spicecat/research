library(MASS)
library(tree)
data(Boston)
set.seed(1)
# tree.boston <- tree(medv ~ ., data = Boston, subset = train)
# summary(tree.boston)
# plot(tree.boston)
# text(tree.boston, pretty = 0)

train_split <- function(dataset, target, frac = 1.) {
  sample_indices <- sample(seq_len(nrow(dataset)), nrow(dataset) / 2)
  train <- dataset[sample_indices, ]
  return(train)
}

tree <- function(dataset, target, features, threshold) {
  num <- dim(dataset)[2]
  b_var <- NULL
  b_rule <- NULL
  b_mse <- NULL
  for (feature in features) {
    a <- paste(feature, "_sorted", sep = "")
    dataset <- cbind(dataset, data.frame(sort(dataset[, feature])))
    colnames(dataset) <- append(names(dataset)[seq_along(names(dataset)) - 1], a)
    b <- paste(feature, "_midpoints", sep = "")
    c <- 0
    for (x in 1:(length(dataset[, a]) - 1)) {
      c <- append(c, ((dataset[x, a] + dataset[x + 1, a]) / 2))
    }
    dataset <- cbind(dataset, c)
    colnames(dataset) <- append(names(dataset)[seq_along(names(dataset)) - 1], b)
    dataset <- cbind(dataset, dataset[order(dataset[, feature]), target])
    d <- paste(feature, "_sorted_dependent", sep = "")
    colnames(dataset) <- append(names(dataset)[seq_along(names(dataset)) - 1], d)
    e <- 0
    for (y in 2:length(dataset[, a])) {
      midpt <- dataset[y, b]
      estimate_below <- mean(dataset[1:(y - 1), d])
      estimate_above <- mean(dataset[y:length(dataset[, d]), d])
      sse1 <- 0
      for (z1 in dataset[1:(y - 1), d]) {
        sse1 <- sse1 + (z1 - estimate_below) * (z1 - estimate_below)
      }

      sse2 <- 0
      for (z2 in dataset[y:length(dataset[, d]), d]) {
        sse2 <- sse2 + (z2 - estimate_above) * (z2 - estimate_above)
      }
      mse <- (sse1 + sse2) / length(dataset[, d])
      e <- append(e, mse)
    }
    f <- paste(feature, "_mse_midpoint", sep = "")
    dataset <- cbind(dataset, e)
    colnames(dataset) <- append(names(dataset)[seq_along(names(dataset)) - 1], f)
    b_var <- append(b_var, feature)
    ranked_rules <- dataset[order(dataset[, f]), ]
    b_rule <- append(b_rule, ranked_rules[2, b])
    b_mse <- append(b_mse, ranked_rules[2, f])
  }
  mse_for_rules <- data.frame(b_var, b_rule, b_mse)
  mse_for_rules_sorted <- mse_for_rules[order(mse_for_rules[, "b_mse"]), ]
  g <- paste(mse_for_rules_sorted[1, "b_var"], " <= ", sep = "")
  h <- paste(g, mse_for_rules_sorted[1, "b_rule"], sep = "")
  if (length(subset(dataset, eval(parse(text = b_var)) < b_rule)[, target]) > threshold) {
    print("if less than")
    print(h)
    print("then")
    tree(subset(dataset, eval(parse(text = b_var)) < b_rule)[, 1:num], target, features, threshold, seq_along(subset(dataset, eval(parse(text = b_var)) < b_rule)[, target]))
  } else {
    print("if less than")
    print(h)
    print("estimate is")
    print(mean(subset(dataset, eval(parse(text = b_var)) < b_rule)[, target]))
  }
  if (length(subset(dataset, eval(parse(text = b_var)) > b_rule)[, target]) > threshold) {
    print("if greater than")
    print(h)
    print("then")
    tree(subset(dataset, eval(parse(text = b_var)) > b_rule)[, 1:num], target, features, threshold, seq_along(subset(dataset, eval(parse(text = b_var)) > b_rule)[, target]))
  } else {
    print("if greater than")
    print(h)
    print("estimate is")
    print(mean(subset(dataset, eval(parse(text = b_var)) > b_rule)[, target]))
  }
}
train <- train_split(Boston, "medv")
tree(train, "medv", c("crim"), 300)
