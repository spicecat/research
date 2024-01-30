import pandas as pd
import graphviz

boston = pd.read_csv('boston.csv')


def train_split(data, target, features=None, frac=0.5):
    if features is None:
        features = data.columns
    train = data.sample(frac=frac)
    return train.drop(target, axis=1).loc[:, features], train[target]


class TerminalNode():
    def __init__(self, node_data):
        self.outcomes = node_data['outcomes']
        self.samples = len(self.outcomes)
        self.score = node_data.get('score', None)
        self.value = node_data.get('value', None)
        self.depth = node_data['depth']

    def __str__(self):
        return (
            f'''{self.depth*'  '}{self.depth} score={self.score:.3f}, samples={
                self.samples}, value={self.value:.3f}'''
        )

    def str(self):
        return (
            f'''score={self.score:.3f}
samples={self.samples}
value={self.value:.3f}'''
        )


class DecisionNode(TerminalNode):
    def __init__(self, rule, left, right, node_data):
        super().__init__(node_data)
        self.feature = rule['feature']
        self.threshold = rule['threshold']
        self.left = left
        self.right = right
        self.depth = node_data['depth']

    def __str__(self):
        return (
            f'''{self.depth*'  '}{self.depth} [{self.feature} < {self.threshold:.3f}] score={self.score:.3f}, samples={self.samples}, value={self.value:.3f}
{self.left}
{self.right}'''
        )

    def str(self):
        return (
            f'''[{self.feature} < {self.threshold:.3f}]
score={self.score:.3f}
samples={self.samples}
value={self.value:.3f}'''
        )


class DecisionTreeRegressor():
    def __init__(self, *, min_samples_leaf=20):
        self.root = None
        self.min_samples_leaf = min_samples_leaf

    def _weighted_average_of_mse(self, splits):
        def _mean_squared_error(dataset):
            target_column = dataset.iloc[:, -1]
            average = self._value(dataset)
            return sum((target_column-average) ** 2)
        n = sum(map(len, splits))
        weight = 0
        for dataset in splits:
            size = len(dataset)
            weight += _mean_squared_error(dataset)*(size/n)
        return weight

    def _value(self, data):
        return data.iloc[:, -1].mean()

    def _score(self, data):
        return self._weighted_average_of_mse([data])

    def _get_split(self, data, depth):
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
        rule = {'feature': b_feature, 'threshold': b_threshold}
        left = {'outcomes': b_groups[0], 'depth': depth+1}
        right = {'outcomes': b_groups[1], 'depth': depth+1}
        node_data = {'outcomes': data, 'score': b_score,
                     'value': self._value(data), 'depth': depth}
        return DecisionNode(rule, TerminalNode(left),
                            TerminalNode(right), node_data)

    def _split(self, node):
        left, right = node.left.outcomes, node.right.outcomes
        # process left child
        if len(left) > self.min_samples_leaf:
            node.left = self._get_split(left, node.depth+1)
            self._split(node.left)
        else:
            node.left.score = self._score(left)
            node.left.value = self._value(left)
        # process right child
        if len(right) > self.min_samples_leaf:
            node.right = self._get_split(right, node.depth+1)
            self._split(node.right)
        else:
            node.right.score = self._score(right)
            node.right.value = self._value(right)

    def fit(self, x, y):
        data = pd.concat([x, y], axis=1)
        print(data)
        root = self._get_split(data, 0)
        self._split(root)
        self.root = root
        return root

    def render(self):
        graph = graphviz.Digraph()
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node is None:
                continue
            graph.node(str(hash(node)), node.str())
            if isinstance(node, DecisionNode):
                queue.append(node.left)  # type: ignore
                queue.append(node.right)  # type: ignore
                graph.edge(str(hash(node)), str(hash(node.left)))
                graph.edge(str(hash(node)), str(hash(node.right)))
        graph.render('tree', view=True)


included_rows = [505, 324, 167, 129, 418, 471,
                 299, 270, 466, 187, 307, 481,  85, 277, 362]
boston = boston.loc[included_rows]

x_train, y_train = train_split(boston, 'Price', ["CRIM", "ZN", "INDUS"])
regressor = DecisionTreeRegressor(min_samples_leaf=2)
regressor.fit(x_train, y_train)
# print(x_train)
print(regressor.root)

regressor.render()
