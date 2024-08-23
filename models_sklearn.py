import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from typing import Literal

import matplotlib.pyplot as plt


class MLP(MLPRegressor):
    pass


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
        self.ensemble = RandomForestRegressor(num_trees)

    def _concat_tree(self, X):
        return np.hstack((X, np.column_stack([e.predict(X) for e in self.ensemble.estimators_])))

    def fit(self, X: np.ndarray, y):
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
        self.ensemble = RandomForestRegressor(num_trees)

    def _concat_tree(self, activations):
        tree_outputs = np.column_stack(
            [e.predict(activations[0]) for e in self.ensemble.estimators_])
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


def evaluate_model(model, X, y, n=1):
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    import time

    data = []

    for random_state in range(n):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)

        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        train_time = end_time - start_time

        start_time = time.time()
        predictions = model.predict(X_test)
        end_time = time.time()
        comp_time = end_time - start_time

        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        n_iter = model.n_iter_

        data.append([r2, mae, mse, train_time, comp_time, n_iter])

    df = pd.DataFrame(
        data, columns=['r2', 'mae', 'mse', 'train_time', 'comp_time', 'n_iter'])

    return df


def plot_loss(model, title='Loss Curve'):
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    import pandas as pd
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler

    def best_estimator(model, param_grid, X, y):
        search = GridSearchCV(model, param_grid)
        search.fit(X, y)
        return search.best_estimator_

    import numpy as np
    np.random.seed(42)

    # Load the Boston dataset
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22,  # type: ignore
                         header=None)  # type: ignore
    X = np.hstack([raw_df.values[::2, :-1], raw_df.values[1::2, :2]])
    y = raw_df.values[1::2, 2].reshape(-1, 1)

    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y).ravel()

    param_grid = {
        'max_iter': [10000],
        'learning_rate': ['constant'],
        'learning_rate_init': [1e-2],
        'tol': [1e-4],
        'early_stopping': [True]
    }

    mlp = MLP(10, max_iter=1000, early_stopping=True)
    fonn1 = best_estimator(FONN1(5, (10,)), param_grid, X, y)
    fonn2 = FONN2(5, (15,), max_iter=1000, early_stopping=True)
    treenn1 = TREENN1((10,), max_iter=1000, early_stopping=True)
    treenn2 = TREENN2((10,), max_iter=1000, early_stopping=True)

    print(evaluate_model(mlp, X, y))
    print(evaluate_model(fonn1, X, y))

    plot_loss(mlp, 'MLP')
    plot_loss(fonn1, 'FONN1')
    # plot_loss(fonn2, 'FONN2')
    # plot_loss(treenn1, 'TREENN1')
    # plot_loss(treenn2, 'TREENN2')
