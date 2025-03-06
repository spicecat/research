"""Main script for running Tree Neural Network experiments."""

import time

import pandas as pd
from config.settings import MODEL_PARAMS, TRAIN_TEST_SPLIT
from data.data_loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from models.forest_models import FONN1, FONN2, FONN3
from models.tree_models import TREENN1, TREENN2, TREENN3


def main():
    # Load data
    X, y = DataLoader.load_boston_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=42
    )

    # Results storage
    results = []

    def evaluate_model(
        name,
        model_func,
        epochs=MODEL_PARAMS["max_iter"],
        learning_rate=MODEL_PARAMS["learning_rate_init"],
        **kwargs,
    ):
        """Evaluates a given model and stores the results."""
        start_time = time.time()
        model = model_func(**kwargs, X_train=X_train, y_train=y_train)
        model.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)
        predictions = model.forward(X_test)
        end_time = time.time()

        results.append(
            {
                "Model": name,
                "R² Score": r2_score(y_test, predictions),
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mean_squared_error(y_test, predictions),
                "Time (s)": end_time - start_time,
            }
        )

    # Base parameters for all models
    base_params = {
        "input_dim": X_train.shape[1],
        "hidden_dim": MODEL_PARAMS["hidden_dim"],
        "output_dim": 1,
    }

    # Evaluate tree models
    for name, model in [
        ("TREENN1", TREENN1),
        ("TREENN2", TREENN2),
        ("TREENN3", TREENN3),
    ]:
        evaluate_model(name, model, **base_params)

    # Evaluate forest models
    forest_params = {**base_params, "num_trees": MODEL_PARAMS["num_trees"]}
    for name, model in [
        ("FONN1", FONN1),
        ("FONN2", FONN2),
        ("FONN3", FONN3),
    ]:
        evaluate_model(name, model, **forest_params)

    # Evaluate PureMLP
    start_time = time.time()
    mlp = MLPRegressor(
        hidden_layer_sizes=(MODEL_PARAMS["hidden_dim"],),
        max_iter=MODEL_PARAMS["max_iter"],
        learning_rate_init=MODEL_PARAMS["learning_rate_init"],
        random_state=MODEL_PARAMS["random_state"],
        n_iter_no_change=MODEL_PARAMS["max_iter"],
    )
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    end_time = time.time()

    results.append(
        {
            "Model": "PureMLP",
            "R² Score": r2_score(y_test, predictions),
            "MAE": mean_absolute_error(y_test, predictions),
            "MSE": mean_squared_error(y_test, predictions),
            "Time (s)": end_time - start_time,
        }
    )

    # Save results
    results_df = pd.DataFrame(results)
    output_file = f'output/model_results_{MODEL_PARAMS["max_iter"]}_{TRAIN_TEST_SPLIT}_{time.strftime("%F_%T")}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(results_df)
    print(mlp.n_iter_)


if __name__ == "__main__":
    main()
