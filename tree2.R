library(MASS)
library(tree)
data(Boston)
set.seed(1)
# tree.boston <- tree(medv ~ ., data = Boston, subset = train)
# summary(tree.boston)
# plot(tree.boston)
# text(tree.boston, pretty = 0)

train_split <- function(dataset, frac = 1.) {
  sample_indices <- sample(seq_len(nrow(dataset)), frac * nrow(dataset))
  train <- dataset[sample_indices, ]
  return(train)
}

get_split <- function(dataset) {

}

# mse_function <- function(dataset, y) {
#   # Calculate the estimate below and above the y-th value
#   estimate_below <- mean(dataset[1:(y - 1), d])
#   estimate_above <- mean(dataset[y:rows, d])

#   # Calculate the SSE for the values below and above the y-th value
#   sse1 <- sum((dataset[1:(y - 1), d] - estimate_below)^2)
#   sse2 <- 0
#   for (z2 in dataset[y:rows, d]) {
#     sse2 <- sse2 + (z2 - estimate_above) * (z2 - estimate_above)
#   }

#   # Calculate the MSE
#   mse <- (sse1 + sse2) / rows

#   # Return the MSE
#   return(mse)
# }

tree <- function(dataset, target, features, threshold) {
  rows <- dim(dataset)[1]
  columns <- dim(dataset)[2]
  b_var <- NULL
  b_rule <- NULL
  b_mse <- NULL
  for (feature in features) {
    a <- paste(feature, "_sorted", sep = "")
    dataset <- cbind(dataset, data.frame(sort(dataset[, feature])))
    names(dataset)[ncol(dataset)] <- a

    b <- paste(feature, "_midpoints", sep = "")
    c <- 0
    for (x in 2:rows) {
      c <- append(c, ((dataset[x - 1, a] + dataset[x, a]) / 2))
    }
    dataset <- cbind(dataset, c)
    names(dataset)[ncol(dataset)] <- b

    dataset <- cbind(dataset, dataset[order(dataset[, feature]), target])
    d <- paste(feature, "_sorted_target", sep = "")
    names(dataset)[ncol(dataset)] <- d

    e <- 0
    for (y in 2:rows) {
      estimate_below <- mean(dataset[1:(y - 1), d])
      estimate_above <- mean(dataset[y:rows, d])

      sse1 <- sum((dataset[1:(y - 1), d] - estimate_below)^2)
      sse2 <- sum((dataset[y:rows, d] - estimate_above)^2)

      mse <- (sse1 + sse2) / rows
      e <- append(e, mse)
    }

    f <- paste(feature, "_mse_midpoint", sep = "")
    dataset <- cbind(dataset, e)
    names(dataset)[ncol(dataset)] <- f

    b_var <- append(b_var, feature)
    ranked_rules <- dataset[order(dataset[, f]), ]
    b_rule <- append(b_rule, ranked_rules[2, b])
    b_mse <- append(b_mse, ranked_rules[2, f])

  }
  mse_for_rules <- data.frame(b_var, b_rule, b_mse)
  mse_for_rules_sorted <- mse_for_rules[order(mse_for_rules[, "b_mse"]), ]
  print(mse_for_rules)

  g <- paste(mse_for_rules_sorted[1, "b_var"], " <= ", sep = "")
  h <- paste(g, mse_for_rules_sorted[1, "b_rule"], sep = "")
  if (length(subset(dataset, eval(parse(text = b_var)) < b_rule)[, target]) > threshold) {
    print("if less than")
    print(h)
    print("then")
    tree(subset(dataset, eval(parse(text = b_var)) < b_rule)[, 1:columns], target, features, threshold, seq_along(subset(dataset, eval(parse(text = b_var)) < b_rule)[, target]))
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
    tree(subset(dataset, eval(parse(text = b_var)) > b_rule)[, 1:columns], target, features, threshold, seq_along(subset(dataset, eval(parse(text = b_var)) > b_rule)[, target]))
  } else {
    print("if greater than")
    print(h)
    print("estimate is")
    print(mean(subset(dataset, eval(parse(text = b_var)) > b_rule)[, target]))
  }
}
included_rows <- c(
  505, 324, 167, 129, 418, 471,
  299, 270, 466, 187, 307, 481, 85, 277, 362
) + 1
Boston <- Boston[included_rows, ]
train <- Boston
# train <- train_split(Boston, 1.)
# print(train)
tree(train, "medv", c("crim", "zn", "indus"), 300)
