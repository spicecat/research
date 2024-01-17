library(MASS)
library(tree)
data(Boston)
set.seed(1)
train <- sample(1:nrow(Boston), nrow(Boston) / 2)
tree.boston <- tree(medv ~ ., data = Boston, subset = train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston, pretty = 0)

tree <- function(dataset, dependent_variable, explanatory_variables, threshold, train){
  num <- dim(dataset)[2]
  dataset <- dataset[train,]
  best_var <- c()
  best_rule <- c()
  best_mse <- c()
  for (explan_var in explanatory_variables){
    a <- paste(explan_var, "_sorted", sep="")
    dataset <- cbind(dataset, data.frame(sort(dataset[,explan_var])))
    colnames(dataset) <- append(names(dataset)[1:length(names(dataset))-1], a)
    b <- paste(explan_var, "_midpoints", sep="")
    c <- c(0)
    for(x in 1:(length(dataset[,a]) -1)) {
      c <- append(c, ((dataset[x,a] + dataset[x+1,a]) / 2))
    }
    dataset <- cbind(dataset, c)
    colnames(dataset) <- append(names(dataset)[1:length(names(dataset))-1], b)
    dataset <- cbind(dataset, dataset[order(dataset[,explan_var]), dependent_variable])
    d <- paste(explan_var, "_sorted_dependent", sep="")
    colnames(dataset) <- append(names(dataset)[1:length(names(dataset))-1], d)
    e <- c(0)
    for(y in 2:length(dataset[,a])){
      midpt <- dataset[y, b]
      estimate_below <- mean(dataset[1:(y-1), d])
      estimate_above <- mean(dataset[y:length(dataset[,d]), d])
      sse1 <- 0
      for(z1 in dataset[1:(y-1), d]) {
        sse1 <- sse1 + (z1 - estimate_below) * (z1 - estimate_below)
      }
      
      sse2 <- 0
      for(z2 in dataset[y:length(dataset[,d]), d]) {
        sse2 <- sse2 + (z2 - estimate_above) * (z2 - estimate_above)
      }
      mse <- (sse1 + sse2) / length(dataset[,d])
      e <- append(e, mse)
    }
    f <- paste(explan_var, "_mse_midpoint", sep="")
    dataset <- cbind(dataset, e)
    colnames(dataset) <- append(names(dataset)[1:length(names(dataset))-1], f)
    best_var <- append(best_var, explan_var)
    ranked_rules <- dataset[order(dataset[,f]),]
    best_rule <- append(best_rule, ranked_rules[2,b])
    best_mse <- append(best_mse, ranked_rules[2,f])
  }
  mse_for_rules <- data.frame(best_var, best_rule, best_mse)
  mse_for_rules_sorted <- mse_for_rules[order(mse_for_rules[,"best_mse"]),]
  g <- paste(mse_for_rules_sorted[1,"best_var"], " <= ", sep="")
  h <- paste(g, mse_for_rules_sorted[1, "best_rule"], sep="")
  if( length(subset(dataset, eval(parse(text = best_var)) < best_rule)[,dependent_variable]) > threshold) {
    print("if less than")
    print(h)
    print("then")
    tree(subset(dataset, eval(parse(text = best_var)) < best_rule)[,1:num], dependent_variable, explanatory_variables, threshold, 1:length(subset(dataset, eval(parse(text = best_var)) < best_rule)[,dependent_variable]))
  } else {
    print("if less than")
    print(h)
    print("estimate is")
    print(mean(subset(dataset, eval(parse(text = best_var)) < best_rule)[,dependent_variable]))
  }
  if( length(subset(dataset, eval(parse(text = best_var)) > best_rule)[,dependent_variable]) > threshold) {
    print("if greater than")
    print(h)
    print("then")
    tree(subset(dataset, eval(parse(text = best_var)) > best_rule)[,1:num], dependent_variable, explanatory_variables, threshold, 1:length(subset(dataset, eval(parse(text = best_var)) > best_rule)[,dependent_variable]))
  } else {
    print("if greater than")
    print(h)
    print("estimate is")
    print(mean(subset(dataset, eval(parse(text = best_var)) > best_rule)[,dependent_variable]))
  }
}
tree(Boston, "medv", c("crim"), 300, 1:506)
