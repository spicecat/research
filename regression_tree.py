from typing import Optional, Tuple
from queue import PriorityQueue
import pandas as pd
from graphviz import Digraph
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

boston = pd.read_csv('data/boston.csv')


class Rule:
    def __init__(self, feature: str, threshold: float):
        self.feature = feature
        self.threshold = threshold

    def groups(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''Split data into two groups according to the rule.'''

        return data[data[self.feature] < self.threshold], data[data[self.feature] > self.threshold]

    def __str__(self) -> str:
        return (
            f'''[{self.feature} < {self.threshold:.3f}]'''
        )


class TerminalNode:
    def __init__(self, outcomes: pd.DataFrame, depth: int):
        self.outcomes = outcomes
        self.samples = len(self.outcomes)
        self.value = self._value(outcomes)
        self.score = self._score(outcomes)
        self.depth = depth
        self._split = None
        self._label = None
        self._impurity_reduction = None

    def _value(self, data: pd.DataFrame) -> float:
        return data.iloc[:, -1].mean()

    def _score(self, data: pd.DataFrame) -> float:
        return self._weighted_average_of_rss((data,))

    def _weighted_average_of_rss(self, splits: Tuple[pd.DataFrame, ...]) -> float:
        def _residual_sum_of_squares(data: pd.DataFrame) -> float:
            target_column = data.iloc[:, -1]
            average = self._value(data)
            return sum((target_column-average) ** 2)
        splits = tuple(filter(lambda split: len(split) > 0, splits))
        n = sum(map(len, splits))
        weight = 0
        for data in splits:
            size = len(data)
            weight += _residual_sum_of_squares(data)*(size/n)
        return weight

    def get_split(self) -> 'DecisionNode':
        if self._split is None:
            data = self.outcomes.copy()
            features = data.columns[:-1]
            b_feature = features[0]
            b_threshold = 0
            b_score = float('inf')
            for feature in features:
                data.sort_values(feature, inplace=True)
                for row_i in range(1, len(data)):
                    if data.iloc[row_i-1][feature] == data.iloc[row_i][feature]:
                        continue
                    left = data[:row_i]
                    right = data[row_i:]
                    score = self._weighted_average_of_rss((left, right))
                    if score < b_score:
                        b_feature = feature
                        b_threshold = (data.iloc[row_i-1][feature] +
                                       data.iloc[row_i][feature])/2
                        b_score = score
            rule = Rule(b_feature, b_threshold)
            self._split = DecisionNode(
                self.outcomes,
                self.depth,
                rule
            )
        return self._split

    def get_impurity_reduction(self):
        if self._impurity_reduction is None:
            split = self.get_split()
            left = split.left
            right = split.right
            self._impurity_reduction = self.samples * self.score - (
                left.samples*left.score+right.samples+right.score
            )
        return self._impurity_reduction

    def split(self):
        self.__class__ = DecisionNode
        self.__dict__ = self.get_split().__dict__

    def get_label(self):
        return (
            f'''score={self.score:.3f}
samples={self.samples}
value={self.value:.3f}'''
        )

    def __lt__(self, other: 'TerminalNode') -> bool:
        return self.get_impurity_reduction() > other.get_impurity_reduction()

    def __str__(self) -> str:
        return (
            f'''{self.depth*'  '}{self.depth} score={self.score:.3f}, samples={
                self.samples}, value={self.value:.3f}'''
        )


class DecisionNode(Rule, TerminalNode):
    def __init__(self, outcomes, depth, rule):
        Rule.__init__(self, rule.feature, rule.threshold)
        TerminalNode.__init__(self, outcomes, depth)
        left, right = self.groups(outcomes)
        self.left = TerminalNode(left, depth+1)
        self.right = TerminalNode(right, depth+1)

    def get_label(self):
        if self._label is None:
            self._label = (
                f'''[{self.feature} < {self.threshold:.3f}]
score={self.score:.3f}
samples={self.samples}
value={self.value:.3f}'''
            )
        return self._label

    def __str__(self):
        return (
            f'''{TerminalNode.__str__(self)} {Rule.__str__(self)}
{self.left}
{self.right}'''
        )


class DecisionTreeRegressor(BaseEstimator):
    '''A decision tree regressor.'''

    def __init__(self, *, min_samples_split=20, min_samples_leaf=1, max_leaf_nodes=None):
        self.min_samples_split: int = min_samples_split
        self.min_samples_leaf: int = min_samples_leaf
        self.max_leaf_nodes: Optional[int] = max_leaf_nodes
        self.root: Optional[TerminalNode] = None

    def _nodes(self, node=None, nodes=None):
        if node is None:
            return self._nodes(self.root, nodes)
        if nodes is None:
            return self._nodes(node, [{
                'id': hash(node),
                'parent': None,
                'node': node
            }])
        if isinstance(node, DecisionNode):
            nodes.append({
                'parent': node,
                'node': node.left
            })
            nodes.append({
                'parent': node,
                'node': node.right
            })
            self._nodes(node.left, nodes)
            self._nodes(node.right, nodes)
        return nodes

    def predict(self, row: pd.Series) -> float:
        if self.root is None:
            return 0
        if not isinstance(self.root, DecisionNode):
            return self.root.value
        node: DecisionNode | TerminalNode = self.root
        while isinstance(node, DecisionNode):
            node = node.left if row[node.feature] < node.threshold else node.right
        return node.value

    def mean_squared_error(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        # print(test)
        test = pd.concat([x_test, y_test], axis=1)
        print(test)
        error_total: float = sum(
            map(lambda row: (row[1].iloc[-1] - self.predict(row[1]))**2, test.iterrows()))
        print(list(
            map(lambda row: (row[1].iloc[-1] - self.predict(row[1]))**2, test.iterrows())))
        return error_total/len(test)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> 'TerminalNode':
        '''Build a decision tree regressor from the training set (X, y).'''
        data = pd.concat([x, y], axis=1)
        root = TerminalNode(data, 0)
        pq = PriorityQueue()
        pq.put(root)

        terminal_nodes = 1
        while not pq.empty() \
                and (self.max_leaf_nodes is None or terminal_nodes < self.max_leaf_nodes):
            node = pq.get()
            if node.samples >= self.min_samples_split:
                split = node.get_split()
                if split.left.samples >= self.min_samples_leaf \
                        and split.right.samples >= self.min_samples_leaf:
                    node.split()
                    pq.put(split.left)
                    pq.put(split.right)
                    terminal_nodes += 1
        self.root = root
        return self.root

    def render(self):
        '''Plot a decision tree.'''

        graph = Digraph()
        nodes = self._nodes()
        for node in nodes:
            node_id = str(hash(node['node']))
            parent_id = str(hash(node['parent']))
            graph.node(node_id, node['node'].get_label())
            if node['parent'] is not None:
                graph.edge(parent_id, node_id)
        graph.render('tree', view=True)

    def summarize(self):
        if self.root is not None:
            nodes = self._nodes()
            decision_nodes = [node['node'] for node in nodes if isinstance(
                node['node'], DecisionNode)]
            terminal_nodes = [node['node'] for node in nodes if not isinstance(
                node['node'], DecisionNode)]
            features = set(map(lambda node: node.feature, decision_nodes))

            error_total = sum(map(
                lambda terminal_node: terminal_node.score * terminal_node.samples,
                terminal_nodes
            ))

            print(f'Variables used:\n{features}')
            print(f'Number of terminal nodes: {len(terminal_nodes)}')
            if error_total:
                print(
                    f'Residual mean deviance: \
{error_total/(self.root.samples - len(terminal_nodes)):.3f}\
 = {error_total:.3f} / ({self.root.samples} - {len(terminal_nodes)})'
                )

# def k_fold_mse(regressor, x: pd.DataFrame, y: pd.Series, k: int):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
#     regressor.fit(x_train, y_train)
#     return regressor.mean_squared_error(x_test, y_test)


included_rows = [505, 324, 167, 129, 418, 471,
                 299, 270, 466, 187, 307, 481,  85, 277, 362]
boston = boston.loc[included_rows]
included_columns = ['CRIM', 'ZN', 'INDUS', 'Price']
boston = boston.loc[:, included_columns]

dataset = boston
X = dataset.drop('Price', axis=1)
y = dataset['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

regressor = DecisionTreeRegressor(
    min_samples_split=2,
    max_leaf_nodes=20
)

regressor.fit(X_train, y_train)
print(regressor.root)
regressor.summarize()

# Predict the outcome for the test row
# test_row = pd.Series(
#     [0.04741, 0, 11.93, 11.9],
#     index=['CRIM', 'ZN', 'INDUS', 'Price']
# )
# print(regressor.predict(test_row))

print(regressor.mean_squared_error(X_test, y_test))

regressor.render()


# cv = KFold(n_splits=5, shuffle=True, random_state=1)
# scores = cross_val_score(regressor, X, y, scoring=scoring, cv=cv, n_jobs=-1)
