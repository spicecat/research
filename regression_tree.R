if (!require("igraph")) {
  install.packages("igraph")
}
if (!require("caret")) {
  install.packages("caret")
}
if (!require("collections")) {
  install.packages("collections")
}
library(igraph)
library(MASS)
library(caret)
library(R6)
library(collections)
# boston <- read.csv("boston.csv")
boston <- Boston

test_train_split <- function(df, train_size = NULL, test_size = NULL) {
  if (is.null(train_size)) {
    if (is.null(test_size)) {
      test_size <- 0.25
    }
    train_size <- 1 - test_size
  } else if (is.null(test_size)) {
    test_size <- 1 - train_size
  }
  train_n <- nrow(df) * train_size
  test_n <- nrow(df) * test_size
  indices <- sample(seq_len(nrow(df)), size = train_n + test_n)
  train_indices <- indices[1:train_n]
  test_indices <- indices[(train_n + 1):(train_n + test_n + 1)]
  train_set <- df[train_indices, ]
  test_set <- df[test_indices, ]
  list(train = train_set, test = test_set)
}

Rule <- R6Class( # nolint
  "Rule",
  public = list(
    feature = NULL,
    threshold = NULL,
    initialize = function(feature, threshold) {
      self$feature <- feature
      self$threshold <- threshold
    },
    evaluate = function(row) {
      row[self$feature] < self$threshold
    },
    get_label = function() {
      sprintf("[%s < %0.3f]", self$feature, self$threshold)
    },
    print = function() {
      print(self$get_label())
    }
  )
)

Node <- R6Class( # nolint
  "Node",
  public = list(
    outcomes = NULL,
    samples = NULL,
    value = NULL,
    score = NULL,
    depth = NULL,
    rule = NULL,
    left = NULL,
    right = NULL,
    initialize = function(outcomes, depth) {
      self$outcomes <- outcomes
      self$samples <- nrow(self$outcomes)
      self$value <- private$get_value(outcomes)
      self$score <- private$get_score(outcomes)
      self$depth <- depth
    },
    get_split = function() {
      if (is.null(private$next_split)) {
        dataset <- self$outcomes
        b_feature <- NULL
        b_threshold <- NULL
        b_score <- Inf
        features <- names(dataset)[-ncol(dataset)]
        n <- nrow(dataset)
        for (feature in features) {
          dataset <- dataset[order(dataset[, feature]), ]
          for (row_i in 2:n) {
            if (dataset[row_i - 1, feature] == dataset[row_i, feature]) {
              next
            }
            left <- dataset[1:(row_i - 1), ]
            right <- dataset[row_i:n, ]
            score <- private$weighted_average_of_mse(list(left, right))
            if (score < b_score) {
              b_feature <- feature
              b_threshold <- mean(dataset[(row_i - 1):row_i, feature])
              b_score <- score
            }
          }
        }
        private$next_split <- Node$new(
          outcomes = self$outcomes,
          depth = self$depth
        )
        rule <- Rule$new(
          feature = b_feature,
          threshold = b_threshold
        )
        private$next_split$rule <- rule
        groups <- split(dataset, apply(dataset, 1, rule$evaluate))
        private$next_split$left <- Node$new(
          outcomes = groups[[2]],
          depth = self$depth + 1
        )
        private$next_split$right <- Node$new(
          outcomes = groups[[1]],
          depth = self$depth + 1
        )
      }
      private$next_split
    },
    get_impurity_reduction = function() {
      next_split <- self$get_split()
      left <- next_split$left
      right <- next_split$right
      self$score * self$samples - (
        left$samples * left$score + right$samples * right$score
      )
    },
    split = function() {
      next_split <- self$get_split()
      self$rule <- next_split$rule
      self$left <- next_split$left
      self$right <- next_split$right
    },
    as.data.frame = function() {
      data.frame(
        samples = self$samples,
        score = self$score,
        value = self$value,
        depth = self$depth,
        feature = ifelse(is.null(self$rule), NA, self$rule$feature),
        threshold = ifelse(is.null(self$rule), NA, self$rule$threshold)
      )
    },
    get_label = function() {
      label <- sprintf(
        "score=%0.3f\nsamples=%d\nvalue=%0.3f",
        self$score, self$samples, self$value
      )
      if (!is.null(self$rule)) {
        label <- paste0(self$rule$get_label(), "\n", label)
      }
      label
    },
    print = function() {
      padding <- paste(rep("  ", self$depth), collapse = "")
      cat(sprintf("%s%d ", padding, self$depth))
      if (!is.null(self$rule)) {
        cat(self$rule$get_label())
      }
      cat(sprintf(
        " score=%0.3f, samples=%d, value=%0.3f\n",
        self$score, self$samples, self$value
      ))
      if (!is.null(self$rule)) {
        print(self$left)
        print(self$right)
      }
    }
  ),
  private = list(
    next_split = NULL,
    get_value = function(dataset) {
      mean(dataset[, ncol(dataset)])
    },
    get_score = function(dataset) {
      private$weighted_average_of_mse(list(dataset))
    },
    weighted_average_of_mse = function(splits) {
      sum_squared_error <- function(dataset) {
        actual <- dataset[, ncol(dataset)]
        average <- private$get_value(dataset)
        sum((actual - average)^2)
      }
      n <- sum(sapply(splits, nrow))
      sum(sapply(
        splits,
        function(dataset) sum_squared_error(dataset) * (nrow(dataset) / n)
      ))
    }
  )
)

DecisionTreeRegressor <- R6Class( # nolint
  "decision_tree_regressor",
  public = list(
    root = NULL,
    min_samples_split = NULL,
    min_samples_leaf = NULL,
    max_leaf_nodes = NULL,
    initialize = function(min_samples_split = 2,
                          min_samples_leaf = 1,
                          max_leaf_nodes = Inf) {
      self$min_samples_split <- min_samples_split
      self$min_samples_leaf <- min_samples_leaf
      self$max_leaf_nodes <- max_leaf_nodes
    },
    fit = function(formula, dataset) {
      dataset <- model.frame(formula, dataset)
      dataset <- aggregate(formula, dataset, mean)
      root <- Node$new(outcomes = dataset, depth = 0)
      if (root$samples >= self$min_samples_split) {
        pq <- PriorityQueue()
        pq$push(root)
        terminal_nodes <- 1
        while (pq$size() > 0 && terminal_nodes < self$max_leaf_nodes) {
          node <- pq$pop()
          if (node$samples >= self$min_samples_split) {
            split <- node$get_split()
            left <- split$left
            right <- split$right
            if (all(left$samples, right$samples >= self$min_samples_leaf)) {
              node$split()
              if (left$samples >= self$min_samples_split) {
                pq$push(left, priority = left$get_impurity_reduction())
              }
              if (right$samples >= self$min_samples_split) {
                pq$push(right, priority = right$get_impurity_reduction())
              }
              terminal_nodes <- terminal_nodes + 1
            }
          }
        }
      }
      self$root <- root
      self$root
    },
    predict = function(row) {
      node <- self$root
      while (!is.null(node$rule)) {
        if (node$rule$evaluate(row)) {
          node <- node$left
        } else {
          node <- node$right
        }
      }
      node$value
    },
    mean_squared_error = function(test, target = NULL) {
      if (is.null(target)) {
        actual <- test[, ncol(test)]
      } else {
        actual <- test[, target]
      }
      predictions <- apply(test, 1, function(row) self$predict(row))
      mean((actual - predictions)^2)
    },
    nodes_df = function(node, id, parent = NA) {
      df <- data.frame(
        id = id,
        parent = parent,
        node = node$as.data.frame(),
        label = node$get_label()
      )
      if (!is.null(node$rule)) {
        df <- rbind(df, self$nodes_df(node$left, paste0(id, "L"), id))
        df <- rbind(df, self$nodes_df(node$right, paste0(id, "R"), id))
      }
      df
    },
    render = function() {
      nodes <- self$nodes_df(self$root, "0")
      g <- graph.tree(n = 0, children = 2)
      g <- g + vertices(nodes[, "id"])
      labels <- nodes[, "label"]
      apply(nodes, 1, function(row) {
        if (!is.na(row["parent"])) {
          g <<- g + edge(row["parent"], row["id"])
        }
      })
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
      nodes <- self$nodes_df(self$root, "0")
      terminal_nodes <- nodes[is.na(nodes$node.feature), ]
      cat("Variables used:\n")
      cat(na.omit(unique(nodes$node.feature)))
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

k_fold_mse <- function(model, formula, dataset, k, max_leaf_nodes = 5) {
  folds <- createFolds(seq_len(nrow(dataset)), k = k)
  start_time <- Sys.time()
  mse_values <- sapply(folds, function(fold) {
    train <- dataset[-fold, ]
    test <- dataset[fold, ]
    model(formula, train, test, max_leaf_nodes)
  })
  time <- Sys.time() - start_time
  print(max_leaf_nodes)
  print(time)
  # print(mse_values)
  mean(mse_values)
}

plot_mse <- function(model, formula, dataset, k, max_leaf_nodes) {
  mse_values <- sapply(max_leaf_nodes, function(i) {
    k_fold_mse(model, formula, dataset, k, max_leaf_nodes = i)
  })
  min_mse <- min(mse_values)
  one_sd <- min_mse + sd(mse_values)
  color_mse <- c()
  for (i in 1:length(mse_values)) {
    if (mse_values[i] == min_mse) {
      color_mse <- append(color_mse, "red")
    }
    else {
      color_mse <- append(color_mse, "black")
    }
  }
  node_index <- 0
  for(i in 1:length(mse_values)) {
    if(mse_values[i] <= one_sd) {
      node_index <- i
      break
    }
  }
  color_mse[node_index] <- "blue"
  plot(max_leaf_nodes, mse_values, col=color_mse)
  data.frame(max_leaf_nodes, mse_values)
}

set.seed(1)

included_rows <- c(
  505, 324, 167, 129, 418, 471,
  299, 270, 466, 187, 307, 481, 85, 277, 362
) + 1
included_columns <- c("crim", "zn", "indus", "medv")
# boston <- boston[, included_columns]
# boston <- boston[included_rows, ]

# test_train <- test_train_split(boston, train_size = 0.8)
# train <- test_train$train
# test <- test_train$test

# Fit the decision tree regressor to the training data
# regressor <- DecisionTreeRegressor$new(
#   min_samples_split = 2,
#   max_leaf_nodes = 5000
# )

# regressor$fit(
#   formula = medv ~ .,
#   dataset = train
# )

# Print the decision tree
# print(regressor$root)
# regressor$summarize()
# regressor$render()


# Predict the outcome for the test row
# test_row <- c(
#   crim = 0.04741,
#   zn = 0,
#   indus = 11.93,
#   medv = 11.9
# )
# print(regressor$predict(test_row))

# print("MSE")
# print(regressor$mean_squared_error(test))

model <- function(formula, train, test, max_leaf_nodes) {
  regressor <- DecisionTreeRegressor$new(
    max_leaf_nodes = max_leaf_nodes
  )
  regressor$fit(
    formula = formula,
    dataset = train
  )
  regressor$mean_squared_error(test)
}

# print("k-fold test")
# print(k_fold_mse(
#   model = model,
#   formula = medv ~ .,
#   dataset = boston,
#   k = 5
# ))

tree_generate <- function(dataset, split, formula, k, leaf_node_test_seq) {
  test_train <- test_train_split(dataset, train_size = split)
  train <- test_train$train
  test <- test_train$test
  mse_table <- plot_mse(
    formula = formula,
    model = model,
    dataset = train,
    k = k,
    max_leaf_nodes = leaf_node_test_seq
  )
  print(mse_table)
  one_sd <- min(mse_table$mse_values) + sd(mse_table$mse_values)
  leaf_nodes <- mse_table$max_leaf_nodes[
    which.min(mse_table$mse_values >= one_sd)
  ]
  regressor <- DecisionTreeRegressor$new(
    max_leaf_nodes = leaf_nodes
  )
  regressor$fit(
    formula = formula,
    dataset = train
  )
  print(regressor$mean_squared_error(test))
  print(regressor$root)
  regressor$render()
  regressor$summarize()
  for(i in 1:dim(train)[1]){
    train$tree_predictions[i] <- regressor$predict(train[i,])
  }
  train
}

train <- tree_generate(boston, 0.8, medv ~ ., 5, seq(1, 5, by = 1))
