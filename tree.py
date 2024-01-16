import pandas as pd

included_columns = ['CRIM', 'RM', 'Price']
boston = pd.read_csv('boston.csv').loc[:, included_columns]


def mean_squared_error(target_column):
    average = target_column.mean()
    return ((target_column-average) ** 2).mean()


def weighted_average_of_mse(target_columns):
    weights = 0
    for target_column in target_columns:
        weights += len(target_column)*mean_squared_error(target_column)
    return weights / sum(map(len, target_columns))


def tree(dataset, target, frac=0.5):
    dataset = dataset.sample(frac=frac, replace=False)
    n = len(dataset)
    features = [col for col in dataset.columns if col != target]
    for feature in features:
        dataset.sort_values(feature, inplace=True)
        for row_index in range(1, n):
            print(row_index, feature, weighted_average_of_mse(
                (dataset[target][:row_index], dataset[target][row_index:])))
    return dataset


print(tree(boston, 'Price', 0.01))
