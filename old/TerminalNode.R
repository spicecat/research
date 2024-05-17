# Class for a terminal node
TerminalNode <- R6::R6Class( # nolint: object_name_linter.
  "TerminalNode",
  list(
    value = NULL,
    initialize = function(value) {
      stopifnot(is.numeric(value))
      self$value <- value
    },
    print = function() {
      cat(sprintf("value=%0.3f\n", self$value))
    },
    predict = function(row) {
      self$value
    }
  )
)
