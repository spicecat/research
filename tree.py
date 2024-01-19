import pandas as pd

boston = pd.read_csv('boston.csv')

included_rows = [505, 324, 167, 129, 418, 471,
                 299, 270, 466, 187, 307, 481,  85, 277, 362]
# included_columns = ['CRIM', 'RM', 'Price']
included_columns = ['CRIM', 'Price']
boston = boston.loc[:, included_columns]


def train_split(data, target, frac=1.):
    train = data.sample(frac=frac)
    return train.drop(target, axis=1), train[target]


class DecisionNode:
    def __init__(self, feature, threshold, left, right):
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold

    def __str__(self):
        return f'[{self.feature} < {self.threshold}]\n Left: \n{self.left}\n Right: \n{self.right}'


class TerminalNode:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def __str__(self):
        # return str(self.outcomes)
        return f'samples={len(self.outcomes)}, value={self.outcomes.iloc[:, -1].mean()}'


class DecisionTreeRegressor():
    def __init__(self, *, min_samples_leaf=20):
        self.root = None
        self.min_samples_leaf = min_samples_leaf

    def _mean_squared_error(self, target_column):
        average = target_column.mean()
        return sum((target_column-average) ** 2)

    def _weighted_average_of_mse(self, target_columns):
        n = sum(map(len, target_columns))
        weight = 0
        for target_column in target_columns:
            size = len(target_column)
            weight += self._mean_squared_error(target_column)*(size/n)
        return weight

    def _get_split(self, data):
        data = data.copy()
        b_feature = None
        b_threshold = None
        b_score = float('inf')
        b_groups = ()
        features = data.columns[:-1]
        target = data.columns[-1]
        n = len(data)
        for feature in features:
            data.sort_values(feature, inplace=True)
            for row_index in range(1, n):
                if data.iloc[row_index-1][feature] == data.iloc[row_index][feature]:
                    continue
                left = data[:row_index]
                right = data[row_index:]
                score = self._weighted_average_of_mse(
                    (left[target], right[target]))
                if score < b_score:
                    b_feature = feature
                    b_threshold = (data.iloc[row_index-1][feature] +
                                   data.iloc[row_index][feature])/2
                    b_score = score
                    b_groups = left, right
        return DecisionNode(b_feature, b_threshold, *b_groups)

    def _split(self, node):
        left, right = node.left, node.right
        # process left child
        if len(left) > self.min_samples_leaf:
            node.left = self._get_split(left)
            self._split(node.left)
        else:
            node.left = TerminalNode(left)
        # process right child
        if len(right) > self.min_samples_leaf:
            node.right = self._get_split(right)
            self._split(node.right)
        else:
            node.right = TerminalNode(right)

    def fit(self, x, y):
        data = pd.concat([x, y], axis=1)
        # print(data)
        root = self._get_split(data)
        self._split(root)
        self.root = root
        return root


x_train, y_train = train_split(boston, 'Price')
regressor = DecisionTreeRegressor(min_samples_leaf=300)
regressor.fit(x_train, y_train)
print(regressor.root)
