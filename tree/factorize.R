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
