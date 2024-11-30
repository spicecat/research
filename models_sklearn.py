import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from typing import Literal


class Ensemble(RandomForestRegressor):
    def __init__(
            self,
            n_estimators=100,
            *,
            stack=False,
            criterion: Literal['squared_error', 'absolute_error',
                               'friedman_mse', 'poisson'] = 'squared_error',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )
        self.stack = stack

    def predict(self, X):
        if self.stack:
            return np.column_stack([e.predict(X) for e in self.estimators_])
        return super().predict(X)


class MLP(MLPRegressor):
    pass


class Tree(Ensemble):
    def __init__(
            self,
            *,
            criterion: Literal['squared_error', 'absolute_error', 'friedman_mse',
                               'poisson'] = 'squared_error',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
    ):
        super().__init__(
            n_estimators=1,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )


class FONN1(MLP):
    def __init__(self,
                 num_trees=10,
                 hidden_layer_sizes=(100,),
                 activation: Literal['relu', 'identity',
                                     'logistic', 'tanh'] = 'relu',
                 *,
                 solver: Literal['lbfgs', 'sgd', 'adam'] = 'adam',
                 alpha=0.0001,
                 batch_size="auto",
                 learning_rate: Literal['constant',
                                        'invscaling', 'adaptive'] = "constant",
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=1e-4,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 n_iter_no_change=10,
                 max_fun=15000,
                 ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
        self.num_trees = num_trees
        self.ensemble = Ensemble(num_trees, stack=True)
        # self.scaler_y = StandardScaler()

    def _concat_tree(self, X):
        return np.hstack((self.ensemble.predict(X), X))
        # return np.hstack((scaler_y.transform(self.ensemble.predict(X).reshape(-1, 1).ravel()), X))

    def fit(self, X: np.ndarray, y):
        # self.scaler_y.fit(y.reshape(-1, 1).ravel())
        self.ensemble.fit(X, y)
        return super().fit(self._concat_tree(X), y)

    def predict(self, X: np.ndarray):
        return super().predict(self._concat_tree(X))


class TREENN1(FONN1):
    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation: Literal['relu', 'identity',
                                     'logistic', 'tanh'] = 'relu',
                 *,
                 solver: Literal['lbfgs', 'sgd', 'adam'] = 'adam',
                 alpha=0.0001,
                 batch_size="auto",
                 learning_rate: Literal['constant',
                                        'invscaling', 'adaptive'] = "constant",
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=1e-4,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 n_iter_no_change=10,
                 max_fun=15000,
                 ):
        super().__init__(
            num_trees=1,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )


class FONN2(MLP):
    def __init__(self,
                 num_trees=10,
                 hidden_layer_sizes=(100,),
                 activation: Literal['relu', 'identity',
                                     'logistic', 'tanh'] = 'relu',
                 *,
                 solver: Literal['lbfgs', 'sgd', 'adam'] = 'adam',
                 alpha=0.0001,
                 batch_size="auto",
                 learning_rate: Literal['constant',
                                        'invscaling', 'adaptive'] = "constant",
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=1e-4,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 n_iter_no_change=10,
                 max_fun=15000,
                 ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
        self.num_trees = num_trees
        self.ensemble = Ensemble(num_trees, stack=True)

    def _concat_tree(self, activations):
        tree_outputs = self.ensemble.predict(activations[0])
        activations[-2] = np.hstack((tree_outputs,
                                     activations[-2][:, self.num_trees:]))
        return activations

    def _forward_pass(self, activations):
        activations = super()._forward_pass(activations)  # type: ignore
        activations = self._concat_tree(activations)
        return activations

    def fit(self, X, y):
        self.ensemble.fit(X, y)
        return super().fit(X, y)


class TREENN2(FONN2):
    def __init__(self,
                 hidden_layer_sizes=(100,),
                 activation: Literal['relu', 'identity',
                                     'logistic', 'tanh'] = 'relu',
                 *,
                 solver: Literal['lbfgs', 'sgd', 'adam'] = 'adam',
                 alpha=0.0001,
                 batch_size="auto",
                 learning_rate: Literal['constant',
                                        'invscaling', 'adaptive'] = "constant",
                 learning_rate_init=0.001,
                 power_t=0.5,
                 max_iter=200,
                 shuffle=True,
                 random_state=None,
                 tol=1e-4,
                 verbose=False,
                 warm_start=False,
                 momentum=0.9,
                 nesterovs_momentum=True,
                 early_stopping=False,
                 validation_fraction=0.1,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 n_iter_no_change=10,
                 max_fun=15000,
                 ):
        super().__init__(
            num_trees=1,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )


def plot_loss(model, title='Loss Curve'):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(model.loss_curve_)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title(title)
    ax1.grid(True)

    ax2.plot(model.loss_curve_)
    ax2.set_yscale('log')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss')
    ax2.set_title(title)
    ax2.grid(True)

    fig.tight_layout()
    fig.show()


if __name__ == '__main__':

    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    import numpy as np
    np.random.seed(42)

    # Load the Boston dataset
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22,  # type: ignore
                         header=None)  # type: ignore
    X = np.hstack([raw_df.values[::2, :-1], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2].reshape(-1, 1)

    X = X.astype(np.float64)
    y = y.astype(np.float64)

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y).ravel()

    models = {}

    hn = 5
    ti = 5
    th = 5

    models[f'MLP {hn}'] = MLP(hn)
    models[f'FONN1 {ti} {hn}'] = FONN1(ti, ti+hn)
    models[f'FONN2 {th} {hn}'] = FONN2(ti, ti+hn)
    models[f'TREENN1 {hn}'] = TREENN1(hn+1)
    models[f'TREENN2 {hn}'] = TREENN2(hn+1)

    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X, y)
        plot_loss(model, title=f'{name} Loss Curve')

    a = input()
