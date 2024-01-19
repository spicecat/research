import pandas as pd

boston = pd.read_csv('boston.csv')

included_rows = [505, 324, 167, 129, 418, 471,
                 299, 270, 466, 187, 307, 481,  85, 277, 362]
included_columns = ['CRIM', 'RM', 'TAX', 'Price']
boston = boston.loc[included_rows, included_columns]


def train_split(data, target, frac=1.):
    train = data.sample(frac=frac)
    return train.drop(target, axis=1), train[target]


class TerminalNode():
    def __init__(self, outcomes, score=None, value=None):
        self.outcomes = outcomes
        self.samples = len(outcomes)
        self.score = score
        self.value = value

    def __str__(self):
        # return str(self.outcomes)
        return f'score={self.score:.3f}, samples={self.samples}, value={self.value:.3f}'


class DecisionNode(TerminalNode):
    def __init__(self, feature, threshold, left, right, outcomes, score, value):
        super().__init__(outcomes, score, value)
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold

    def __str__(self):
        return f'[{self.feature} < {self.threshold:.3f}]\n Left: \n{self.left}\n Right: \n{self.right}'


class DecisionTreeRegressor():
    def __init__(self, *, min_samples_leaf=20):
        self.root = None
        self.min_samples_leaf = min_samples_leaf

    def _weighted_average_of_mse(self, datasets):
        def _mean_squared_error(dataset):
            target_column = dataset.iloc[:, -1]
            average = self._value(dataset)
            return sum((target_column-average) ** 2)
        n = sum(map(len, datasets))
        weight = 0
        for dataset in datasets:
            size = len(dataset)
            weight += _mean_squared_error(dataset)*(size/n)
        return weight

    def _value(self, data):
        return data.iloc[:, -1].mean()

    def _score(self, data):
        return self._weighted_average_of_mse([data])

    def _get_split(self, data):
        data = data.copy()
        b_feature = None
        b_threshold = None
        b_score = float('inf')
        b_groups = (None, None)
        features = data.columns[:-1]
        n = len(data)
        for feature in features:
            data.sort_values(feature, inplace=True)
            for row_index in range(1, n):
                if data.iloc[row_index-1][feature] == data.iloc[row_index][feature]:
                    continue
                left = data[:row_index]
                right = data[row_index:]
                score = self._weighted_average_of_mse((left, right))
                if score < b_score:
                    b_feature = feature
                    b_threshold = (data.iloc[row_index-1][feature] +
                                   data.iloc[row_index][feature])/2
                    b_score = score
                    b_groups = left, right
        return DecisionNode(b_feature, b_threshold, TerminalNode(b_groups[0]),
                            TerminalNode(b_groups[1]), data, b_score, self._value(data))

    def _split(self, node):
        left, right = node.left.outcomes, node.right.outcomes
        # process left child
        if len(left) > self.min_samples_leaf:
            node.left = self._get_split(left)
            self._split(node.left)
        else:
            node.left.score = self._score(left)
            node.left.value = self._value(left)
        # process right child
        if len(right) > self.min_samples_leaf:
            node.right = self._get_split(right)
            self._split(node.right)
        else:
            node.right.score = self._score(right)
            node.right.value = self._value(right)

    def fit(self, x, y):
        data = pd.concat([x, y], axis=1)
        # print(data)
        root = self._get_split(data)
        self._split(root)
        self.root = root
        return root


x_train, y_train = train_split(boston, 'Price')
regressor = DecisionTreeRegressor(min_samples_leaf=5)
regressor.fit(x_train, y_train)
print(boston)
print(regressor.root)
