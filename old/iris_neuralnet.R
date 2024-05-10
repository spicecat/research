# LOAD NECESSARY LIBRARIES
packages <- c("keras", "neuralnet", "tidyverse", "tensorflow")
installed_packages <- packages %in% rownames(installed.packages())
install.packages(packages[!installed_packages])
lapply(packages, library, character.only = TRUE)

iris <- iris %>% mutate_if(is.character, as.factor)
print(summary(iris))
set.seed(245)
data_rows <- floor(0.80 * nrow(iris))
train_indices <- sample(c(seq_len(nrow(iris))), data_rows)
train_data <- iris[train_indices, ]
test_data <- iris[-train_indices, ]
model <- neuralnet(
  Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = train_data,
  hidden = c(4, 2),
  linear.output = FALSE
)
plot(model, rep = "best")
pred <- predict(model, test_data)
labels <- c("setosa", "versicolor", "virginca")
prediction_label <- data.frame(max.col(pred)) %>%
  mutate(pred = labels[max.col.pred.]) %>%
  select(2) %>%
  unlist()

# table(test_data$Species, prediction_label)
check <- as.numeric(test_data$Species) == max.col(pred)
accuracy <- (sum(check) / nrow(test_data)) * 100
print(accuracy)
