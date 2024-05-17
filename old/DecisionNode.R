# Class for a decision node
DecisionNode <- R6::R6Class( # nolint: object_name_linter.
  "DecisionNode",
  list(
    feature = NULL,
    threshold = NULL,
    initialize = function(feature, threshold) {
      stopifnot(is.character(feature), is.numeric(threshold))
      self$feature <- feature
      self$threshold <- threshold
    },
    print = function() {
      rule_label <- sprintf("[%s < %0.3f]", self$feature, self$threshold)
      print(rule_label)
    },
    eval = function(row) {
      row[self$feature] < self$threshold
    }
  )
)
