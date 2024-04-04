# Install and load required packages
packages <- c("methods", "MASS")
lapply(packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
})

classes <- c("tree/DecisionTreeRegressor.R")
lapply(classes, source)

test_train_split <- function(data, size = 0.75) {
  sample.int(n = nrow(data), size = floor(size * nrow(data)))
}

set.seed(1)

# Load datasets
boston <- Boston
alcohol <- read.csv("data/student-mat.csv")

# Split dataset into train and test sets
sample <- sample.int(n = nrow(boston), size = floor(0.75 * nrow(boston)))
train <- boston[sample, ]
test <- boston[-sample, ]

# Fit the decision tree regressor to the training data
regressor <- DecisionTreeRegressor$new(
  max_leaf_nodes = 100L
)
regressor$fit(
  formula = medv ~ .,
  data = train
)
print(regressor)

# # Load decision tree classes
# classes <- c("tree/DecisionNode.R", "tree/TerminalNode.R")
# lapply(classes, source)

# criteria <- c(
#   squared_error = function(y, y_hat) (y - y_hat)^2
# )

# # Class for the decision tree regressor
# DecisionTreeRegressor <- R6::R6Class( # nolint: object_name_linter.
#   "DecisionTreeRegressor",
#   list(
#     criterion = NULL, # Function to measure the quality of a split
#     min_samples_split = NULL, # Minimum samples to be a leaf node
#     max_leaf_nodes = NULL, # Maximum leaf nodes in the tree
#     nodes = c(), # Array of nodes
#     initialize = function(criterion = "squared_error",
#                           min_samples_split = 2L,
#                           max_leaf_nodes = .Machine$integer.max) {
#       stopifnot(is.function(criterion) | criterion %in% names(criteria))
#       stopifnot(is.numeric(min_samples_split))
#       stopifnot(is.numeric(max_leaf_nodes))
#       self$criterion <- ifelse(
#         is.function(criterion),
#         criterion,
#         criteria[[criterion]]
#       )
#       self$min_samples_split <- min_samples_split
#       self$max_leaf_nodes <- max_leaf_nodes
#     },
#     print = function() {

#     },
#     add_node = function(index) {
#       b_feature <- NA_character_
#       b_threshold <- NA_real_
#       b_score <- Inf
#       samples <- private$samples[[index]]
#       features <- names(samples)[-1]
#       n <- nrow(samples)
#       for (feature in features) {
#         sorted <- samples[order(samples[, feature]), ]
#         # results <- sapply(2:nrow(samples), private$impurity, sorted, feature)
#         for (row_i in 2:n) {
#           if (sorted[row_i - 1, feature] == sorted[row_i, feature]) next
#           threshold <- mean(sorted[(row_i - 1):row_i, feature])
#           left <- sorted[1:(row_i - 1), ]
#           right <- sorted[row_i:n, ]
#           self$nodes[[index]] <- DecisionNode$new(feature, threshold)
#           self$nodes[[2 * index]] <- TerminalNode$new(mean(left[[1]]))
#           self$nodes[[2 * index + 1]] <- TerminalNode$new(mean(right[[1]]))
#           score <- private$impurity(index)
#           if (score < b_score) {
#             b_feature <- feature
#             b_threshold <- threshold
#             b_score <- score
#           }
#         }
#       }
#       print(b_feature)
#       print(b_threshold)
#       print(b_score)
#       # if (is.na(b_feature)) {
#       #   return(self$nodes[[index]])
#       # }
#       # decision_node <- DecisionNode$new(b_feature, b_threshold)
#       # self$nodes[[index]] <- decision_node
#       # self$nodes[[2 * index]] <- TerminalNode$new(mean(left[[1]]))
#       # private$samples[[2 * index]] <- subset(
#       #   samples,
#       #   apply(samples, 1, decision_node$eval)
#       # )
#       # self$nodes[[2 * index + 1]] <- TerminalNode$new(mean(right[[1]]))
#       # private$samples[[2 * index + 1]] <- subset(
#       #   samples,
#       #   !apply(samples, 1, decision_node$eval)
#       # )
#       # private$terminal_nodes <- private$terminal_nodes + 1
#       # print(self$nodes)
#       # print(private$samples)
#       # decision_node
#     },
#     predict = function(x, index = 1) {
#       stopifnot(private$is_fitted)
#       if (inherits(self$nodes[[index]], "DecisionNode")) {
#         self$predict(
#           x,
#           if (self$nodes[[index]]$eval(x)) {
#             2 * index
#           } else {
#             2 * index + 1
#           }
#         )
#       } else {
#         self$nodes[[index]]$value
#       }
#     },
#     fit = function(formula, data) {
#       # Fit the decision tree regressor to the dataset
#       df <- model.frame(formula, data)
#       private$terminal_nodes <- 1
#       private$is_fitted <- TRUE
#       self$nodes[[1]] <- TerminalNode$new(mean(df[[1]]))
#       private$samples[[1]] <- df
#       private$pq$push(1)
#       while (private$pq$size() > 0) {
#         index <- private$pq$pop()
#         if (nrow(private$samples[[index]]) < self$min_samples_split) next
#         self$add_node(index)
#       }
#     }
#   ),
#   list(
#     terminal_nodes = 0,
#     samples = c(),
#     is_fitted = FALSE,
#     pq = collections::PriorityQueue(),
#     impurity = function(index = 1) {
#       predictions <- apply(private$samples[[index]], 1, self$predict, index)
#       sum(self$criterion(predictions, private$samples[[index]][1]))
#     }
#   )
# )
