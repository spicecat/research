# Load decision tree classes
classes <- c("tree/DecisionNode.R", "tree/TerminalNode.R")
lapply(classes, source)

criteria <- c(
  squared_error = function(y, y_hat) (y - y_hat)^2
)

# Class for the decision tree regressor
DecisionTreeRegressor <- R6::R6Class( # nolint: object_name_linter.
  "DecisionTreeRegressor",
  list(
    criterion = NULL, # Function to measure the quality of a split
    min_samples_split = NULL, # Minimum samples to be a leaf node
    max_leaf_nodes = NULL, # Maximum leaf nodes in the tree
    nodes = c(), # Array of nodes
    initialize = function(criterion = "squared_error",
                          min_samples_split = 2L,
                          max_leaf_nodes = .Machine$integer.max) {
      stopifnot(is.function(criterion) | criterion %in% names(criteria))
      stopifnot(is.numeric(min_samples_split))
      stopifnot(is.numeric(max_leaf_nodes))
      self$criterion <- ifelse(
        is.function(criterion),
        criterion,
        criteria[[criterion]]
      )
      self$min_samples_split <- min_samples_split
      self$max_leaf_nodes <- max_leaf_nodes
    },
    print = function(index = 1) {
      if (index <= length(self$nodes) && !is.null(self$nodes[[index]])) {
        padding <- paste(rep("  ", log2(index)), collapse = "")
        cat(sprintf("%s%d ", padding, index))
        print(self$nodes[[index]])
        self$print(2 * index)
        self$print(2 * index + 1)
      }
    },
    add_node = function() {
      split <- private$pq$pop()
      index <- split[[1]]
      feature <- split[[3]]
      threshold <- split[[4]]
      if (!is.na(feature)) {
        self$nodes[[index]] <- DecisionNode$new(feature, threshold)
        samples <- private$samples[[index]]
        splits <- split(samples, self$nodes[[index]]$eval(samples))
        left <- splits$T
        right <- splits$F
        self$nodes[[2 * index]] <- TerminalNode$new(mean(left[[1]]))
        self$nodes[[2 * index + 1]] <- TerminalNode$new(mean(right[[1]]))
        private$samples[[2 * index]] <- left
        private$samples[[2 * index + 1]] <- right
        left_split <- private$get_split(2 * index)
        right_split <- private$get_split(2 * index + 1)
        private$pq$push(left_split, priority = -left_split[[2]])
        private$pq$push(left_split, priority = -right_split[[2]])
        private$terminal_nodes <- private$terminal_nodes + 1
      }
      self$nodes[[index]]
    },
    predict = function(x, index = 1) {
      stopifnot(private$is_fitted)
      if (inherits(self$nodes[[index]], "DecisionNode")) {
        self$predict(
          x,
          if (self$nodes[[index]]$eval(x)) {
            2 * index
          } else {
            2 * index + 1
          }
        )
      } else {
        self$nodes[[index]]$value
      }
    },
    fit = function(formula, data) {
      # Fit the decision tree regressor to the dataset
      df <- model.frame(formula, data)
      private$terminal_nodes <- 1
      private$samples[[1]] <- df
      root <- private$get_split(1)
      private$pq$push(root)
      while (all(
        private$pq$size() > 0,
        private$terminal_nodes < self$max_leaf_nodes
      )) {
        self$add_node()
      }
      private$is_fitted <- TRUE
    }
  ),
  list(
    terminal_nodes = 0,
    samples = c(),
    is_fitted = FALSE,
    pq = collections::PriorityQueue(),
    get_split = function(index) {
      samples <- private$samples[[index]]
      features <- names(samples)[-1]
      n <- nrow(samples)
      b_score <- Inf
      b_feature <- NA_character_
      b_threshold <- NA_real_
      if (nrow(private$samples[[index]]) >= self$min_samples_split) {
        for (feature in features) {
          sorted <- samples[order(samples[, feature]), ]
          for (row_i in 2:n) {
            if (sorted[row_i - 1, feature] == sorted[row_i, feature]) next
            left <- sorted[1:(row_i - 1), ]
            right <- sorted[row_i:n, ]
            score <- sum(self$criterion(left[[1]], mean(left[[1]]))) +
              sum(self$criterion(right[[1]], mean(right[[1]])))
            if (score < b_score) {
              b_score <- score
              b_feature <- feature
              b_threshold <- mean(sorted[(row_i - 1):row_i, feature])
            }
          }
        }
      }
      list(index, b_score, b_feature, b_threshold)
    }
  )
)
