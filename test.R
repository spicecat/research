# install.packages(c("neuralnet", "keras", "tensorflow"), dependencies = T)
library(tidyverse)
library(neuralnet)
iris <- iris %>% mutate_if(is.character, as.factor)
print(summary(iris))
set.seed(245)
data_rows <- floor(0.80 * nrow(iris))
train_indices <- sample(c(seq_len(nrow(iris))), data_rows)
train_data <- iris[train_indices, ]
test_data <- iris[-train_indices, ]
