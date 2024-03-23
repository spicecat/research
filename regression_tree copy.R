# Install and load required packages
packages <- c("collections", "methods", "MASS")
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
  return(data)
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
  test_indices <- indices[(train_n + 1):(train_n + test_n)]
  train_set <- data[train_indices, ]
  test_set <- data[test_indices, ]
  return(list(train = train_set, test = test_set))
}

# Class for a decision rule
setClass(
  "Rule",
  slots = list(
    feature = "character",
    threshold = "numeric"
  ),
  prototype = list(
    feature = NA_character_,
    threshold = NA_real_
  ),
  validity = function(object) {
    # Check feature: Must not be NULL
    if (is.null(object@feature)) {
      return("Invalid 'feature': Cannot be NULL.")
    }
    # Check threshold: Must be numeric
    if (!is.numeric(object@threshold)) {
      return("Invalid 'threshold': Must be numeric.")
    }
    TRUE
  }
)

setMethod("show", "Rule", function(object) {
  # Print the rule
  print(sprintf("[%s < %0.3f]", object@feature, object@threshold))
})
setMethod("str", "Rule", function(object) {
  # Print the rule
  sprintf("[%s < %0.3f]", object@feature, object@threshold)
})


# Class for a terminal node
TerminalNode <- setClass( # nolint: object_name_linter.
  "TerminalNode",
  slots = list(
    index = "numeric",
    value = "numeric"
  ),
  prototype = list(
    index = 1L,
    value = NA_real_
  ),
  validity = function(object) {
    # Check index: Must be a positive integer
    if (!is.integer(object@index) || object@index <= 0) {
      return("Invalid 'index': Must be a positive integer.")
    }
    # Check value: Must be numeric
    if (!is.numeric(object@value)) {
      return("Invalid 'value': Must be numeric.")
    }
    TRUE
  }
)

setMethod("show", "TerminalNode", function(object) {
  # Print the terminal node
  depth <- floor(log2(object@index))
  padding <- paste(rep("  ", depth), collapse = "")
  cat(sprintf("%s%d ", padding, depth))
  cat(sprintf(
    " value=%0.3f\n",
    object@value
  ))
})
setMethod("str", "TerminalNode", function(object) {
  # Print the terminal node
  depth <- floor(log2(object@index))
  padding <- paste(rep("  ", depth), collapse = "")
  cat(sprintf("%s%d ", padding, depth))
  cat(sprintf(
    " value=%0.3f\n",
    object@value
  ))
})

# Class for a terminal node
DecisionNode <- setClass( # nolint: object_name_linter.
  "DecisionNode",
  contains = "Rule",
  slots = list(
    index = "numeric",
    rule = "ANY",
    left = "ANY",
    right = "ANY"
  ),
  prototype = list(
    index = 1L,
    rule = NULL,
    left = NULL,
    right = NULL
  ),
  validity = function(object) {
    # Check index: Must be a positive integer
    if (!is.integer(object@index) || object@index <= 0) {
      return("Invalid 'index': Must be a positive integer.")
    }
    # Check rule: Must not be NULL
    if (is.null(object@rule)) {
      return("Invalid 'rule': Cannot be NULL.")
    }
    # Check left and right: Must not be NULL
    if (is.null(object@left) || is.null(object@right)) {
      return("Invalid 'left' or 'right': Cannot be NULL.")
    }
    TRUE
  }
)

setMethod("show", "DecisionNode", function(object) {
  # Print the decision node
  depth <- floor(log2(object@index))
  padding <- paste(rep("  ", depth), collapse = "")
  cat(sprintf("%s%d ", padding, depth))
  print(object$rule)
  print(object$left)
  print(object$right)
})
setMethod("str", "DecisionNode", function(object) {
  # Print the decision node
  depth <- floor(log2(object@index))
  padding <- paste(rep("  ", depth), collapse = "")
  cat(sprintf("%s%d ", padding, depth))
  print(object$rule)
  print(object$left)
  print(object$right)
})

# Class for the decision tree regressor
DecisionTreeRegressor <- setClass( # nolint: object_name_linter.
  "DecisionTreeRegressor",
  slots = list(
    min_samples_split = "numeric",
    min_samples_leaf = "numeric",
    max_leaf_nodes = "numeric",
    root = "ANY",
    pq = "ANY",
    nodes = "list",
    outcomes = "list"
  ),
  prototype = list(
    min_samples_split = 2,
    min_samples_leaf = 1,
    max_leaf_nodes = Inf,
    root = NULL,
    pq = PriorityQueue(),
    nodes = list(),
    outcomes = list()
  )
)

setGeneric(
  "regressor.fit",
  function(object, formula, data) standardGeneric("regressor.fit")
)
setMethod(
  "regressor.fit",
  "DecisionTreeRegressor",
  function(object, formula, data) {
    df <- model.frame(formula, data)
    # df <- aggregate(formula, df, mean)
    object@root <- TerminalNode(value = mean(df[, ncol(df)]))
    print(formula)
    print(object@root)
  }
)

setGeneric(
  "regressor.addNodes",
  function(object, n) standardGeneric("regressor.addNodes")
)
setMethod("regressor.addNodes", "DecisionTreeRegressor", function(object, n) {

})

setMethod("predict", "DecisionTreeRegressor", function(object, row) {

})

setGeneric(
  "meanSquaredError",
  function(object, test) standardGeneric("meanSquaredError")
)
setMethod("meanSquaredError", "DecisionTreeRegressor", function(object, test) {

})

setGeneric(
  "regressor.render",
  function(object, test) standardGeneric("regressor.render")
)
setMethod("regressor.render", "DecisionTreeRegressor", function(object) {

})

setMethod("summary", "DecisionTreeRegressor", function(object) {

})

setMethod("show", "DecisionTreeRegressor", function(object) {
  # Print the decision tree
  print(object@root)
})

boston <- Boston
test_train <- test_train_split(boston, train_size = 0.8)
train <- test_train$train
test <- test_train$test

regressor <- DecisionTreeRegressor(
  max_leaf_nodes = 5
)
regressor.fit(regressor, medv ~ ., train)
