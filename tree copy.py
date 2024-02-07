import pandas as pd
from queue import PriorityQueue
from graphviz import Digraph

boston = pd.read_csv('boston.csv')


def train_split(data, target, features=None, frac=0.5):
    '''Split dataset into random train subset.'''

    if features is None:
        features = data.columns
    train = data.sample(frac=frac)
    return train.drop(target, axis=1).loc[:, features], train[target]


class Rule:
    def __init__(self, feature: str, threshold: float):
        self.feature = feature
        self.threshold = threshold

    def groups(self, data: pd.DataFrame) -> tuple:
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
        self.depth = depth
        self.value = self._value(outcomes)
        self.score = self._score(outcomes)
        self._split = None
        self._label = None
        self._impurity_reduction = None

    def _value(self, data: pd.DataFrame) -> float:
        return data.iloc[:, -1].mean()

    def _score(self, data: pd.DataFrame) -> float:
        return self._weighted_average_of_mse((data,))

    def _weighted_average_of_mse(self, splits: tuple[pd.DataFrame, ...]) -> float:
        def _mean_squared_error(data):
            target_column = data.iloc[:, -1]
            average = self._value(data)
            return sum((target_column-average) ** 2)
        splits = tuple(filter(lambda split: len(split) > 0, splits))
        n = sum(map(len, splits))
        weight = 0
        for data in splits:
            size = len(data)
            weight += _mean_squared_error(data)*(size/n)
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
                for row_index in range(1, len(data)):
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
            rule = Rule(b_feature, b_threshold)
            self._split = DecisionNode(
                self.outcomes,
                self.depth,
                rule
            )
        return self._split

    def get_impurity_reduction(self):
        if self._impurity_reduction is None:
            self._impurity_reduction = self.samples * (self.score - self._weighted_average_of_mse(
                self.get_split().groups(self.outcomes)))
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


class DecisionTreeRegressor:
    '''A decision tree regressor.'''

    def __init__(self, *, min_samples_split=20, min_samples_leaf=1, max_leaf_nodes=None):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.root = None

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

    def fit(self, x, y):
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

        if self.root is not None:
            print(f"""Variables used:
{features}
Number of terminal nodes: {len(terminal_nodes)}
Residual mean deviance: {error_total/(self.root.samples - len(terminal_nodes))}
Samples: {self.root.samples}
Sum of squared residuals: {error_total}
""")

    # summarize = function() {
    #   nodes <- nodes_df(root, "0")
    #   terminal_nodes <- nodes[is.na(nodes$node.feature), ]
    #   cat("Variables used:\n")
    #   cat(unique(na.omit(nodes$node.feature)))
    #   cat("\n")
    #   cat(sprintf(
    #     "Number of terminal nodes: %d\n",
    #     nrow(terminal_nodes)
    #   ))
    #   error_total <- sum(
    #     terminal_nodes$node.score * terminal_nodes$node.samples
    #   )
    #   samples <- nodes[1, ]$node.samples
    #   cat(sprintf(
    #     "Residual mean deviance: %0.3f\n",
    #     error_total / (samples - nrow(terminal_nodes))
    #   ))
    #   cat(sprintf(
    #     "Samples: %d\n",
    #     samples
    #   ))
    #   cat(sprintf(
    #     "Sum of squared residuals: %0.3f\n",
    #     error_total
    #   ))
    # }


included_rows = [505, 324, 167, 129, 418, 471,
                 299, 270, 466, 187, 307, 481,  85, 277, 362]
boston = boston.loc[included_rows]

x_train, y_train = train_split(boston, 'Price', ["CRIM", "ZN", "INDUS"], 1.)

dataset = pd.concat([x_train, y_train], axis=1)
# print(a.split)

regressor = DecisionTreeRegressor(min_samples_split=3, max_leaf_nodes=5)
regressor.fit(x_train, y_train)
print(regressor.root)
regressor.summarize()

# regressor.render()
