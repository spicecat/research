# library(MASS)
# library(tree)
# data(Boston)
set.seed(1)
# train <- sample(seq_len(nrow(Boston)), nrow(Boston) / 2)
# tree.boston <- tree(medv ~ ., data = Boston, subset = train)
# summary(tree.boston)
# plot(tree.boston)
# text(tree.boston, pretty = 0)

Boston <- read.csv("boston.csv")

tree <- function(dataset, target, features, min_samples_leaf) {
  rows <- nrow(dataset)
  b_var <- NULL
  b_threshold <- NULL
  b_mse <- NULL
  for (feature in features) {
    feature_sorted <- sort(dataset[, feature])

    feature_midpoint <- NULL
    for (x in 2:rows) {
      feature_midpoint[x] <- (feature_sorted[x - 1] + feature_sorted[x]) / 2
    }

    target_sorted <- dataset[order(dataset[, feature]), target]

    mse <- NULL
    for (y in 2:rows) {
      estimate_below <- mean(target_sorted[1:(y - 1)])
      estimate_above <- mean(target_sorted[y:rows])

      sse1 <- sum((target_sorted[1:(y - 1)] - estimate_below)^2)
      sse2 <- sum((target_sorted[y:rows] - estimate_above)^2)

      mse[y] <- ((sse1 * (y - 1) + sse2 * (rows - y + 1)) / rows)
    }

    b_var <- c(b_var, feature)
    ranked_rules <- order(mse)

    b_threshold <- c(b_threshold, feature_midpoint[ranked_rules[1]])
    b_mse <- c(b_mse, mse[ranked_rules[1]])
  }

  mse_for_rules <- data.frame(b_var, b_threshold, b_mse)
  mse_for_rules_sorted <- mse_for_rules[order(mse_for_rules[, "b_mse"]), ]
  # print(mse_for_rules_sorted)

  var <- mse_for_rules_sorted[1, "b_var"]
  threshold <- mse_for_rules_sorted[1, "b_threshold"]
  left <- dataset[dataset[[var]] < threshold, ]
  right <- dataset[dataset[[var]] > threshold, ]
  # print(var)
  # print(rule)
  # print(left)
  # print(right)


  if (nrow(left) > min_samples_leaf) {
    cat(sprintf("[%s < %0.3f]\n", var, threshold))
    tree(left, target, features, min_samples_leaf)
  } else {
    cat(sprintf("[%s < %0.3f]\nvalue: %0.3f\n", var, threshold, mean(left[, target])))
  }
  if (nrow(right) > min_samples_leaf) {
    cat(sprintf("[%s < %0.3f]\n", var, threshold))
    tree(right, target, features, min_samples_leaf)
  } else {
    cat(sprintf("[%s < %0.3f]\nvalue: %0.3f\n", var, threshold, mean(right[, target])))
  }
}

included_rows <- c(
  505, 324, 167, 129, 418, 471,
  299, 270, 466, 187, 307, 481, 85, 277, 362
) + 1
Boston <- Boston[included_rows, c("CRIM", "ZN", "INDUS", "Price")]
print(Boston)
# tree(Boston, "medv", c("crim", "zn", "indus"), 5, 1:15)
tree(Boston, "Price", c("CRIM", "ZN", "INDUS"), 2)
