if (!require("igraph")) {
  install.packages("igraph")
}
if (!require("caret")) {
  install.packages("caret")
}
library(igraph)
library(MASS)
library(caret)
# boston <- read.csv("boston.csv")
boston <- Boston

# train_test_split <- function(dataset, frac = 0.5) {
#   sample_indices <- sample(nrow(dataset), frac * nrow(dataset))
#   train <- dataset[sample_indices, ]
#   return(train)
# }

test_train_split <- function(df, train_size = NULL, test_size = NULL) {
  if (is.null(train_size)) {
    if (is.null(test_size)) {
      test_size <- 0.25
    }
    train_size <- 1 - test_size
  } else if (is.null(test_size)) {
    test_size <- 1 - train_size
  }
  train_n <- ifelse(train_size < 1, nrow(df) * train_size, train_size)
  test_n <- ifelse(test_size < 1, nrow(df) * test_size, test_size)
  indices <- sample(seq_len(nrow(df)), size = train_n + test_n)
  train_indices <- indices[1:train_n]
  test_indices <- indices[(train_n + 1):(train_n + test_n + 1)]
  train_set <- df[train_indices, ]
  test_set <- df[test_indices, ]
  return(list(train = train_set, test = test_set))
}


terminal_node <- setRefClass("terminal_node",
  fields = list(
    outcomes = "data.frame",
    score = "numeric",
    value = "numeric",
    depth = "numeric"
  ),
  methods = list(
    show = function() {
      samples <- nrow(outcomes)
      padding <- paste(rep("  ", depth), collapse = "")
      cat(sprintf(
        "%s%d score=%0.3f, samples=%d, value=%0.3f\n",
        padding, depth, score, samples, value
      ))
    },
    get_label = function() {
      samples <- nrow(outcomes)
      return(sprintf(
        "score=%0.3f\nsamples=%d\nvalue=%0.3f",
        score, samples, value
      ))
    },
    as.data.frame = function() {
      df <- data.frame(
        # outcomes = outcomes,
        score = score,
        samples = nrow(outcomes),
        value = value,
        depth = depth,
        feature = NA,
        threshold = NA,
        label = get_label()
      )
      return(df)
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
    show = function() {
      samples <- nrow(outcomes)
      padding <- paste(rep("  ", depth), collapse = "")
      cat(sprintf(
        "%s%d [%s < %0.3f] score=%0.3f, samples=%d, value=%0.3f\n",
        padding, depth, feature, threshold, score, samples, value
      ))
      print(left)
      print(right)
    },
    get_label = function() {
      samples <- nrow(outcomes)
      return(sprintf(
        "[%s < %0.3f]\n%s",
        feature, threshold, callSuper()
      ))
    },
    as.data.frame = function() {
      df <- callSuper()
      df$feature <- feature
      df$threshold <- threshold
      df$label <- get_label()
      return(df)
    }
  )
)

decision_tree_regressor <- setRefClass("decision_tree_regressor",
  fields = list(
    root = "decision_node",
    min_samples_split = "numeric",
    min_samples_leaf = "numeric",
    max_leaf_nodes = "numeric"
  ),
  methods = list(
    weighted_average_of_mse = function(splits) {
      sum_squared_error <- function(dataset) {
        actual <- dataset[, ncol(dataset)]
        average <- value(dataset)
        return(sum((actual - average)^2))
      }
      n <- sum(sapply(splits, nrow))
      weight <- 0
      for (dataset in splits) {
        size <- nrow(dataset)
        weight <- weight + (sum_squared_error(dataset) * (size / n))
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
            b_threshold <- (data[row_index - 1, feature] + data[row_index, feature]) / 2
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
      if (nrow(left) >= min_samples_split) {
        node$left <- get_split(left, node$depth + 1)
        split(node$left)
      } else {
        node$left$score <- score(left)
        node$left$value <- value(left)
      }
      # process right child
      if (nrow(right) >= min_samples_split) {
        node$right <- get_split(right, node$depth + 1)
        split(node$right)
      } else {
        node$right$score <- score(right)
        node$right$value <- value(right)
      }
    },
    fit = function(dataset, target = NA, features = c()) {
      if (is.na(target)) {
        target <- names(dataset)[ncol(dataset)]
      }
      if (length(features) == 0) {
        features <- names(dataset)[-which(names(dataset) == target)]
      }
      x <- dataset[, features]
      y <- dataset[[target]]
      data <- cbind(x, y)
      root <<- get_split(data, 0)
      split(root)
      return(root)
    },
    predict = function(row) {
      node <- root
      while (is(node, "decision_node")) {
        if (row[node$feature] < node$threshold) {
          node <- node$left
        } else {
          node <- node$right
        }
      }
      return(node$value)
    },
    mean_squared_error = function(test, target) {
      if (is.na(target)) {
        actual <- test[, ncol(test)]
      } else {
        actual <- test[, target]
      }
      predictions <- apply(test, 1, function(row) predict(row))
      return(mean((actual - predictions)^2))
    },
    nodes_df = function(node, id, parent = NA) {
      df <- data.frame(
        id = id,
        parent = parent,
        node = node$as.data.frame(),
        label = node$get_label()
      )
      if (is(node, "decision_node")) {
        df <- rbind(df, nodes_df(node$left, paste(id, "L"), id))
        df <- rbind(df, nodes_df(node$right, paste(id, "R"), id))
      }
      return(df)
    },
    render = function() {
      nodes <- nodes_df(root, "0")
      g <- graph.tree(n = 0, children = 2)
      labels <- c()
      for (row in seq_len(nrow(nodes))) {
        g <- g + vertices(nodes[row, "id"])
        if (!is.na(nodes[row, "parent"])) {
          g <- g + edge(nodes[row, "parent"], nodes[row, "id"])
        }
        labels <- c(labels, nodes[row, "node.label"])
      }
      l <- layout_as_tree(g, root = "0")
      plot(
        g,
        layout = l,
        vertex.label = labels,
        vertex.shape = "rectangle",
        vertex.size = 35,
        vertex.size2 = 30
      )
    },
    summarize = function() {
      nodes <- nodes_df(root, "0")
      terminal_nodes <- nodes[is.na(nodes$node.feature), ]
      cat("Variables used:\n")
      cat(unique(na.omit(nodes$node.feature)))
      cat("\n")
      cat(sprintf(
        "Number of terminal nodes: %d\n",
        nrow(terminal_nodes)
      ))
      error_total <- sum(
        terminal_nodes$node.score * terminal_nodes$node.samples
      )
      samples <- nodes[1, ]$node.samples
      print(nrow(terminal_nodes))
      cat(sprintf(
        "Residual mean deviance: %0.3f\n",
        error_total / (samples - nrow(terminal_nodes))
      ))
      cat(sprintf(
        "Samples: %d\n",
        samples
      ))
      cat(sprintf(
        "Sum of squared residuals: %0.3f\n",
        error_total
      ))
    }
  )
)


k_fold_mse <- function(dataset, k, min_samples_split, target) {
  dataset <- dataset[sample(nrow(dataset)), ]
  # print(dataset)
  folds <- cut(seq(1, nrow(dataset)), breaks = k, labels = FALSE)
  mse_total <- 0
  for (i in 1:k) {
    test_indexes <- which(folds == i, arr.ind = TRUE)
    test <- dataset[test_indexes, ]
    train <- dataset[-test_indexes, ]
    regressor <- decision_tree_regressor$new(min_samples_split = min_samples_split)
    regressor$fit(train, target)
    mse_total <- mse_total + regressor$mean_squared_error(test, target)
  }
  return(mse_total) / k
}

plot_mse <- function(n, dataset, k, target) {
  mse_values <- 1:n
  for (i in 1:n) {
    temp <- k_fold_mse(boston, 10, i, "medv")
    mse_values[i] <- temp
  }
  plot(1:n, mse_values)
}

included_rows <- c(
  505, 324, 167, 129, 418, 471,
  299, 270, 466, 187, 307, 481, 85, 277, 362
) + 1
included_columns <- c("crim", "zn", "indus", "medv")
boston <- boston[included_rows, included_columns]
# boston <- boston[, included_columns]
# boston <- boston[included_rows, ]

set.seed(1)

test_train <- test_train_split(boston, test_size = 0.65)
train <- test_train$train
test <- test_train$test

# Fit the decision tree regressor to the training data
regressor <- decision_tree_regressor$new(min_samples_split = 2)
# regressor$fit(train, "Price", c("CRIM", "ZN", "INDUS"))
regressor$fit(train)
regressor$fit(train, "medv")


# Print the decision tree
print(regressor$root)

# Predict the outcome for the test row
# test_row <- data.frame(
#   crim = 0.04741,
#   zn = 0,
#   indus = 11.93,
#   medv = 11.9
# )
# print(regressor$predict(test_row[1, ]))

print("MSE")
test <- boston[-as.numeric(rownames(train)), ]
print(regressor$mean_squared_error(test, "medv"))

# print("k-fold test")
# print(k_fold_mse(boston, 10, 5, "medv"))

# print(boston)

# plot_mse(5, boston, 10, "medv")

regressor$summarize()

# regressor$render()
