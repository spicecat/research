library(tree)
library(MASS)

data(Boston)
# Boston <- read.csv("data/boston.csv")

boston <- Boston

# print(boston)

included_rows <- c(
  505, 324, 167, 129, 418, 471,
  299, 270, 466, 187, 307, 481, 85, 277, 362
) + 1
included_columns <- c("crim", "zn", "indus", "medv")
# boston <- boston[included_rows, included_columns]

set.seed(1)
train <- sample(seq_len(nrow(Boston)), nrow(Boston) * 0.5)
print(train)
tree.boston <- tree(medv ~ ., data = Boston[train, ])
print(summary(tree.boston))
plot(tree.boston)
text(tree.boston, pretty = 0)
