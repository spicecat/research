boston <- read.csv("boston.csv")

train_split <- function(dataset, frac = 1) {
  sample_indices <- sample(nrow(dataset), frac * nrow(dataset))
  train <- dataset[sample_indices, ]
  return(train)
}

terminal_node <- setRefClass("terminal_node",
  fields = list(
    outcomes = "data.frame",
    score = "numeric",
    value = "numeric",
    depth = "numeric"
  ),
  methods = list(
    show = function(...) {
      samples <- nrow(outcomes)
      padding <- paste(rep("  ", depth), collapse = "")
      cat(sprintf("%s%d score=%0.3f, samples=%d, value=%0.3f\n", padding, depth, score, samples, value))
    }
  )
)

decision_node <- setRefClass("decision_node",
  contains = "terminal_node",
  fields = list(
    left = "terminal_node",
    right = "terminal_node",
    feature = "character",
    threshold = "numeric",
    depth = "numeric"
  ),
  methods = list(
    show = function(...) {
      samples <- nrow(outcomes)
      padding <- paste(rep("  ", depth), collapse = "")
      cat(sprintf("%s%d [%s < %0.3f] score=%0.3f, samples=%d, value=%0.3f\n", padding, depth, feature, threshold, score, samples, value))
      print(left)
      print(right)
    }
  )
)

decision_tree_regressor <- setRefClass("decision_tree_regressor",
  fields = list(
    root = "decision_node",
    min_samples_leaf = "numeric"
  ),
  methods = list(
    weighted_average_of_mse = function(splits) {
      mean_squared_error <- function(dataset) {
        target_column <- dataset[, ncol(dataset)]
        average <- value(dataset)
        return(sum((target_column - average)^2))
      }
      n <- sum(sapply(splits, nrow))
      weight <- 0
      for (dataset in splits) {
        size <- nrow(dataset)
        weight <- weight + (mean_squared_error(dataset) * (size / n))
      }
      return(weight)
    },
    value = function(data) {
      return(mean(data[, ncol(data)]))
    },
    score = function(data) {
      return(weighted_average_of_mse(list(data)))
    },
    get_split = function(data, depth) {
      b_feature <- NULL
      b_threshold <- NULL
      b_score <- Inf
      b_groups <- c(NULL, NULL)
      features <- names(data)[-ncol(data)]
      n <- nrow(data)
      for (feature in features) {
        data <- data[order(data[, feature]), ]
        for (row_index in 2:n) {
          if (data[row_index - 1, feature] == data[row_index, feature]) {
            next
          }
          left <- data[1:(row_index - 1), ]
          right <- data[row_index:n, ]
          score <- weighted_average_of_mse(list(left, right))
          if (score < b_score) {
            b_feature <- feature
            b_threshold <- (data[row_index - 1, feature] +
              data[row_index, feature]) / 2
            b_score <- score
            b_groups <- list(left, right)
          }
        }
      }
      return(decision_node$new(
        left = terminal_node$new(outcomes = b_groups[[1]], depth = depth + 1),
        right = terminal_node$new(outcomes = b_groups[[2]], depth = depth + 1),
        feature = b_feature,
        threshold = b_threshold,
        outcomes = data,
        score = b_score,
        value = value(data),
        depth = depth
      ))
    },
    split = function(node) {
      left <- node$left$outcomes
      right <- node$right$outcomes
      if (nrow(left) > min_samples_leaf) {
        node$left <- get_split(left, node$depth + 1)
        split(node$left)
      } else {
        node$left$score <- score(left)
        node$left$value <- value(left)
      }
      # process right child
      if (nrow(right) > min_samples_leaf) {
        node$right <- get_split(right, node$depth + 1)
        split(node$right)
      } else {
        node$right$score <- score(right)
        node$right$value <- value(right)
      }
    },
    fit = function(dataset, target, features) {
      x <- dataset[, features]
      y <- dataset[[target]]
      data <- cbind(x, y)
      root <<- get_split(data, 0)
      split(root)
      return(root)
    }
  )
)


included_rows <- c(
  505, 324, 167, 129, 418, 471,
  299, 270, 466, 187, 307, 481, 85, 277, 362
) + 1
boston <- boston[included_rows, ]

# Split the data into training and testing sets
train <- train_split(boston, 1.)

# Fit the decision tree regressor to the training data
regressor <- decision_tree_regressor$new(min_samples_leaf = 2)
regressor$fit(train, "Price", c("CRIM", "ZN", "INDUS"))

# Print the training data
# print(x_train)

# Print the decision tree
print(regressor$root)
