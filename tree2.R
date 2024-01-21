library(MASS)
library(tree)
data(Boston)
set.seed(1)

find_b_rule <- function(dataset, features, threshold) {
  b_var <- NULL
  b_rule <- NULL
  b_mse <- NULL

  for (explan_var in features) {
    a <- paste(explan_var, "_sorted", sep = "")
    dataset <- cbind(dataset, data.frame(sort(dataset[, explan_var])))
    colnames(dataset) <- append(names(dataset)[seq_along(names(dataset)) - 1], a)
    b <- paste(explan_var, "_midpoints", sep = "")
    c <- 0
    for (x in 1:(length(dataset[, a]) - 1)) {
      c <- append(c, ((dataset[x, a] + dataset[x + 1, a]) / 2))
    }
    dataset <- cbind(dataset, c)
    colnames(dataset) <- append(names(dataset)[seq_along(names(dataset)) - 1], b)
    dataset <- cbind(dataset, dataset[order(dataset[, explan_var]), target])
    d <- paste(explan_var, "_sorted_target", sep = "")
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
    f <- paste(explan_var, "_mse_midpoint", sep = "")
    dataset <- cbind(dataset, e)
    colnames(dataset) <- append(names(dataset)[seq_along(names(dataset)) - 1], f)
    b_var <- append(b_var, explan_var)
    ranked_rules <- dataset[order(dataset[, f]), ]
    b_rule <- append(b_rule, ranked_rules[2, b])
    b_mse <- append(b_mse, ranked_rules[2, f])
  }

  return(list(b_var, b_rule, b_mse))
}

print_rule <- function(b_var, b_rule, target) {
  g <- paste(b_var, " <= ", sep = "")
  h <- paste(g, b_rule, sep = "")
  print(paste("if less than"))
  print(h)
  print("then")
}

print_estimate <- function(dataset, b_var, b_rule, target) {
  g <- paste(b_var, " <= ", sep = "")
  h <- paste(g, b_rule, sep = "")
  print(paste("if less than"))
  print(h)
  print(paste("estimate is"))
  print(mean(subset(dataset, eval(parse(text = b_var)) < b_rule)[, target]))
}

tree <- function(dataset, target, features, threshold, train) {
  dataset <- dataset[train, ]

  b_var <- find_b_rule(dataset, features, threshold)$b_var
  b_rule <- find_b_rule(dataset, features, threshold)$b_rule
  b_mse <- find_b_rule(dataset, features, threshold)$b_mse

  if (length(subset(dataset, eval(parse(text = b_var)) < b_rule)[, target]) > threshold) {
    print_rule(b_var, b_rule, target)
    tree(subset(dataset, eval(parse(text = b_var)) < b_rule)[, 1:num], target, features, threshold, seq_along(subset(dataset, eval(parse(text = b_var)) < b_rule)[, target]))
  } else {
    print_estimate(dataset, b_var, b_rule, target)
  }

  if (length(subset(dataset, eval(parse(text = b_var)) > b_rule)[, target]) > threshold) {
    print_rule(b_var, b_rule, target)
    tree(subset(dataset, eval(parse(text = b_var)) > b_rule)[, 1:num], target, features, threshold, seq_along(subset(dataset, eval(parse(text = b_var)) > b_rule)[, target]))
  } else {
    print_estimate(dataset, b_var, b_rule, target)
  }
}
tree(Boston, "medv", c("crim"), 300, 1:506)
