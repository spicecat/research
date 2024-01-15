import pandas as pd

boston = pd.read_csv('boston.csv').drop('Unnamed: 0', axis=1)


def tree(dataset, target, frac=1.):
    dataset = dataset.sample(frac=frac, replace=False)
    features = [col for col in dataset.columns if col != target]
    for col in features:
        dataset.sort_values(col, inplace=True)
        for row in dataset:
            print(234234, row)
    return dataset


print(tree(boston, 'Price', 0.01))
