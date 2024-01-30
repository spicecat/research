library(tree)

Boston <- read.csv("boston.csv")
included_rows <- c(
    505, 324, 167, 129, 418, 471,
    299, 270, 466, 187, 307, 481, 85, 277, 362
) + 1
Boston <- Boston[included_rows, c("CRIM", "ZN", "INDUS", "Price")]

set.seed(1)
# train <- sample(seq_len(nrow(Boston)), nrow(Boston) / 2)
tree.boston <- tree::tree(Price ~ ., data = Boston)
summary(tree.boston)
plot(tree.boston)
text(tree.boston, pretty = 0)
