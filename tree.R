library(MASS)
help(Boston)

tree.boston <- tree(medv~., Boston)
summary(tree.boston)
plot(tree.boston)
text(tree.boston)