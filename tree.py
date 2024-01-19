import pandas as pd

boston = pd.read_csv('boston.csv')

included_rows = [505, 324, 167, 129, 418, 471,
                 299, 270, 466, 187, 307, 481,  85, 277, 362]
included_columns = ['CRIM', 'RM', 'Price']
boston = boston.loc[included_rows, included_columns]


def train_split(dataset, target, frac=1.):
    train = dataset.sample(frac=frac)
    return train.drop(target, axis=1), train[target]


class Node:
    def __init__(self, feature, value, left, right):
        self.left = left
        self.right = right
        self.feature = feature
        self.value = value


class DecisionTreeRegressor():
    def __init__(self, *, min_samples_leaf=20):
        self.root = None
        self.min_samples_leaf = min_samples_leaf

    def _mean_squared_error(self, target_column):
        average = target_column.mean()
        return ((target_column-average) ** 2).mean()

    def _weighted_average_of_mse(self, target_columns):
        n = sum(map(len, target_columns))
        weight = 0
        for target_column in target_columns:
            size = len(target_column)
            weight += self._mean_squared_error(target_column)*(size/n)
            print(4321, size, n, weight)
        return weight

    def _get_split(self, dataset):
        b_feature = None
        b_value = None
        b_score = float('inf')
        b_groups = ()
        features = dataset.columns[:-1]
        target = dataset.columns[-1]
        n = len(dataset)
        for feature in features:
            dataset.sort_values(feature, inplace=True)
            for row_index in range(1, n):
                if dataset.iloc[row_index-1][feature] == dataset.iloc[row_index][feature]:
                    continue
                left = dataset[:row_index]
                right = dataset[row_index:]
                print(111, left)
                print()
                print(222, right)
                # print(111, dataset.loc[row_index, feature])
                score = self._weighted_average_of_mse(
                    (left[target], right[target]))
                print('s', score)
                if score < b_score:
                    b_feature = feature
                    b_value = (dataset[feature][row_index-1] +
                               dataset[feature][row_index])/2
                    b_score = score
                    b_groups = left, right
        return Node(b_feature, b_value, *b_groups)

    def _split(self, node):
        left, right = node.left, node.right
        print(left, right)
        return

    def fit(self, x, y):
        dataset = pd.concat([x, y], axis=1)
        print(dataset)
        root = self._get_split(dataset)
        # self._split(root)
        self.root = root
        return root


x_train, y_train = train_split(boston, 'Price')
regressor = DecisionTreeRegressor(min_samples_leaf=5)
regressor.fit(x_train, y_train)
