# Install and load required packages
packages <- c("igraph", "MASS", "caret", "R6", "collections", "matrixStats")
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

set.seed(1)
debug <- TRUE

# Converts categorical features to numeric values
factorize <- function(data) {
  for (col in names(data)) {
    if (!is.numeric(data[[col]])) {
      data[[col]] <- as.integer(factor(
        data[[col]],
        levels = sort(unique(data[[col]]))
      ))
    }
  }
  data
}


# Split dataset into train and test sets
test_train_split <- function(data, train_size = NULL, test_size = NULL) {
  if (is.null(train_size)) {
    if (is.null(test_size)) {
      test_size <- 0.25
    }
    train_size <- 1 - test_size
  } else if (is.null(test_size)) {
    test_size <- 1 - train_size
  }
  train_n <- nrow(data) * train_size
  test_n <- nrow(data) * test_size
  indices <- sample(seq_len(nrow(data)), size = train_n + test_n)
  train_indices <- indices[1:train_n]
  test_indices <- indices[(train_n + 1):(train_n + test_n + 1)]
  train_set <- data[train_indices, ]
  test_set <- data[test_indices, ]
  list(train = train_set, test = test_set)
}

# Class for a decision rule
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

# Class for a decision tree node
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
      # Find the best split for the current node
      if (is.null(private$next_split)) {
        df <- self$outcomes
        b_feature <- NULL
        b_threshold <- NULL
        b_score <- Inf
        features <- names(df)[-ncol(df)]
        n <- nrow(df)
        for (feature in features) {
          df <- df[order(df[, feature]), ]
          for (row_i in 2:n) {
            if (df[row_i - 1, feature] == df[row_i, feature]) {
              next
            }
            left <- df[1:(row_i - 1), ]
            right <- df[row_i:n, ]
            score <- private$weighted_average_of_mse(list(left, right))
            if (score < b_score) {
              b_feature <- feature
              b_threshold <- mean(df[(row_i - 1):row_i, feature])
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
        groups <- split(df, apply(df, 1, rule$evaluate))
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
      # Calculate the impurity reduction by from splitting the current node
      next_split <- self$get_split()
      left <- next_split$left
      right <- next_split$right
      self$score * self$samples - (
        left$samples * left$score + right$samples * right$score
      )
    },
    split = function() {
      # Split the current node by the best split
      next_split <- self$get_split()
      self$rule <- next_split$rule
      self$left <- next_split$left
      self$right <- next_split$right
    },
    as.data.frame = function() {
      # Get node information as a data frame
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
      # Get the formatted label for the node
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
      # Print the node information
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
    get_value = function(data) {
      # Calculate the value (mean) of target column of the node dataset
      mean(data[, ncol(data)])
    },
    get_score = function(data) {
      # Calculate the score (MSE) for the dataset
      private$weighted_average_of_mse(list(data))
    },
    weighted_average_of_mse = function(splits) {
      # Calculate the weighted average of scores (MSE) for the splits
      sum_squared_error <- function(df) {
        actual <- df[, ncol(df)]
        average <- private$get_value(df)
        sum((actual - average)^2)
      }
      n <- sum(sapply(splits, nrow))
      sum(sapply(
        splits,
        function(df) sum_squared_error(df) * (nrow(df) / n)
      ))
    }
  )
)

# Class for the decision tree regressor
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
    fit = function(formula, data) {
      # Fit the decision tree regressor to the dataset
      df <- model.frame(formula, data)
      df <- aggregate(formula, df, mean)
      root <- Node$new(outcomes = df, depth = 0)
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
      # Predict the target variable for a given row
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
      # Calculate the mean squared error for the test set
      if (is.null(target)) {
        actual <- test[, ncol(test)]
      } else {
        jj
        actual <- test[, target]
      }
      predictions <- apply(test, 1, function(row) self$predict(row))
      mean((actual - predictions)^2)
    },
    nodes_df = function(node, id, parent = NA) {
      # Convert the decision tree to a data frame
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
      # Render the decision tree as a graph
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
      # Print summary statistics of the decision tree
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

# Get test estimates from trained decision tree regressor at max_leaf_nodes
regression_model <- function(formula, train, max_leaf_nodes) {
  regressor <- DecisionTreeRegressor$new(
    max_leaf_nodes = max_leaf_nodes
  )
  regressor$fit(
    formula = formula,
    data = train
  )
  regressor
  # regressor$mean_squared_error(test)
}

# Get k-fold cross-validation score for a model
k_fold_mse <- function(model, formula, data, k, max_leaf_nodes = 5) {
  folds <- caret::createFolds(seq_len(nrow(data)), k = k)
  start_time <- Sys.time()
  mse_values <- sapply(folds, function(fold) {
    train <- data[-fold, ]
    test <- data[fold, ]
    regressor <- model(formula, train, max_leaf_nodes)
    regressor$mean_squared_error(test)
  })
  if (debug) {
    time <- Sys.time() - start_time
    cat(sprintf(
      "Max leaf nodes: %d\n",
      max_leaf_nodes
    ))
    print(time)
  }
  mean(mse_values)
}

# Plot k-fold cross-validation scores for a model by max_leaf_nodes
plot_mse <- function(model, formula, data, k, max_leaf_nodes_seq) {
  mse_values <- sapply(max_leaf_nodes_seq, function(i) {
    k_fold_mse(model, formula, data, k, max_leaf_nodes = i)
  })
  min_mse <- min(mse_values)
  one_sd <- min_mse + sd(mse_values)

  optimal_index <- which.max(mse_values <= one_sd)
  color_mse <- ifelse(mse_values == min_mse, "red", "black")

  color_mse[optimal_index] <- "blue"

  plot(max_leaf_nodes_seq, mse_values, col = color_mse)
  data.frame(max_leaf_nodes_seq, mse_values)
}

# Get model with max_leaf_nodes one standard deviation above minimum MSE
get_optimal_tree <- function(model, formula, data, k, max_leaf_nodes_seq) {
  mse_values <- sapply(max_leaf_nodes_seq, function(i) {
    k_fold_mse(model, formula, data, k, max_leaf_nodes = i)
  })
  min_mse <- min(mse_values)
  one_sd <- min_mse + sd(mse_values)
  optimal_index <- which.max(mse_values <= one_sd)
  optimal_leaf_nodes <- max_leaf_nodes_seq[optimal_index]

  regressor <- DecisionTreeRegressor$new(
    max_leaf_nodes = optimal_leaf_nodes
  )
  regressor$fit(
    formula = formula,
    data = data
  )

  regressor
}

# Load Boston housing dataset
# boston <- read.csv("data/boston.csv")
boston <- Boston
alcohol <- read.csv("data/student-mat.csv")
alcohol <- factorize(alcohol)

# Use simplified subset of Boston dataset
included_rows <- c(
  505, 324, 167, 129, 418, 471,
  299, 270, 466, 187, 307, 481, 85, 277, 362
) + 1
included_columns <- c("crim", "zn", "indus", "medv")
boston <- boston[, included_columns]
# # boston <- boston[included_rows, ]

# Split dataset into train and test sets
# test_train <- test_train_split(boston, train_size = 0.8)
# train <- test_train$train
# test <- test_train$test

# # Fit the decision tree regressor to the training data
# regressor <- DecisionTreeRegressor$new(
#   min_samples_split = 2,
#   max_leaf_nodes = 50
# )
# regressor$fit(
#   formula = medv ~ .,
#   data = train
# )

# # Print the decision tree
# print(regressor$root)
# regressor$summarize()
# regressor$render()


# # Predict the outcome for the test row
# test_row <- c(
#   crim = 0.04741,
#   zn = 0,
#   indus = 11.93,
#   medv = 11.9
# )
# print(regressor$predict(test_row))

# # print("MSE")
# print(regressor$mean_squared_error(test))

# # Test k-fold cross validation score
# print("k-fold test")
# print(k_fold_mse(
#   model = regression_model,
#   formula = medv ~ .,
#   data = boston,
#   k = 5
# ))

# Plot k-fold cross validation scores
# max_leaf_nodes_seq <- seq(2, 400, by = 2)
# max_leaf_nodes_seq <- seq(2, 10, by = 2)
# plot <- plot_mse(
#   model = regression_model,
#   formula = medv ~ .,
#   data = boston,
#   k = 5,
#   max_leaf_nodes_seq = max_leaf_nodes_seq
# )
# print(plot)
max_leaf_nodes_seq <- seq(2, 10, by = 2)
plot <- plot_mse(
  model = regression_model,
  formula = G3 ~ .,
  data = alcohol,
  k = 5,
  max_leaf_nodes_seq = max_leaf_nodes_seq
)
print(plot)

# # Generate an optimal decision tree regressor
# append_predictions <- function(data, split, formula, k, max_leaf_nodes_seq) {
#   test_train <- test_train_split(data, train_size = split)
#   train <- test_train$train
#   test <- test_train$test
#   mse_table <- plot_mse(
#     formula = formula,
#     model = regression_model,
#     data = train,
#     k = k,
#     max_leaf_nodes = max_leaf_nodes_seq
#   )
#   one_sd <- min(mse_table$mse_values) + sd(mse_table$mse_values)
#   leaf_nodes <- mse_table$max_leaf_nodes[
#     which.min(mse_table$mse_values >= one_sd)
#   ]
#   regressor <- DecisionTreeRegressor$new(
#     max_leaf_nodes = leaf_nodes
#   )
#   regressor$fit(
#     formula = formula,
#     data = train
#   )
#   if (debug) {
#     print(mse_table)
#     print(regressor$mean_squared_error(test))
#     print(regressor$root)
#     regressor$render()
#     regressor$summarize()
#   }
#   train$tree_predictions <- apply(train, 1, regressor$predict)
#   train
# }

# bootstrap_columns <- function(data, n, split, formula, k, max_leaf_nodes_seq) {
#   # For each bootstrap sample
#   for (i in 1:n) {
#     # Generate a bootstrap sample
#     bootstrap_sample <- data[sample(nrow(data), replace = TRUE), ]

#     # Fit the decision tree regressor to the bootstrap sample
#     bootstrap_model <- append_predictions(
#       bootstrap_sample,
#       split = split,
#       formula = formula,
#       k = k,
#       leaf_node_test_seq = max_leaf_nodes_seq
#     )

#     # Add the predictions as a new column to the dataset
#     data[[paste0("tree_predictions_", i)]] <- bootstrap_model$tree_predictions
#   }

#   # Return the dataset with the new columns
#   data
# }


# train <- append_predictions(boston, 0.8, medv ~ ., 5, seq(2, 10, by = 2))

# print(train)










# nonlin <- function(x, deriv = FALSE) {
#   if (deriv == TRUE) {
#     y <- 1 / (1 + exp(-x))
#     return(y * (1 - y))
#   }
#   1 / (1 + exp(-x))
# }

# # input data/output labels
# X <- matrix(c(
#   0, 1, 1,
#   1, 1, 0,
#   1, 0, 0,
#   1, 0, 1
# ), nrow = 4, byrow = TRUE)

# X <- train

# y <- matrix(c(1, 1, 0, 0), nrow = 4)

# y <- train$medv
# # setting random seed for reproducing
# set.seed(1)

# # initialize weights
# w0 <- 2 * matrix(runif(12), nrow = 3) - 1
# w1 <- 2 * matrix(runif(4), nrow = 4) - 1
# # hyperparameters
# learning_rate <- 0.01
# epochs <- 10
# batch_size <- 100
# iterations_per_epoch <- 10000

# total_iterations <- 0

# # training loop
# for (epoch in 1:epochs) {
#   # random batch selection
#   indices <- sample(nrow(X))
#   X_shuffled <- X[indices, ]
#   y_shuffled <- y[indices, ]

#   for (batch_start in seq(1, nrow(X), by = batch_size)) {
#     # Select a random batch
#     batch_end <- min(nrow(X), batch_start + batch_size - 1)

#     X_batch <- X_shuffled[batch_start:batch_end, ]
#     y_batch <- y_shuffled[batch_start:batch_end]

#     # forward prop
#     a0 <- X_batch
#     a1 <- nonlin(a0 %*% w0)
#     a2 <- nonlin(a1 %*% w1)

#     # backprop
#     a2_error <- (1 / 2) * (y_batch - a2)^2
#     a2_delta <- a2_error * nonlin(a2, deriv = TRUE)
#     a1_error <- a2_delta %*% t(w1)
#     a1_delta <- a1_error * nonlin(a1, deriv = TRUE)

#     # update weights
#     w1 <- w1 + t(a1) %*% a2_delta * learning_rate
#     w0 <- w0 + t(a0) %*% a1_delta * learning_rate

#     total_iterations <- total_iterations + 1
#     if (total_iterations == iterations_per_epoch) {
#       break # exit the loop after reaching the desired iterations
#     }
#   }

#   # error val after each epoch
#   print(paste("Epoch", epoch, ", Error:", mean(abs(a2_error))))
# }

# # final weights
# print("Final weights:")
# print("w0:")
# print(w0)
# print("w1:")
# print(w1)
