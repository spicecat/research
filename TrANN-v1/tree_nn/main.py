"""Main script for running Tree Neural Network experiments."""

import time

from config.settings import MODEL_PARAMS, TRAIN_TEST_SPLIT
from data.data_loader import DataLoader
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from utils.evaluation import ModelEvaluator
from utils.scaler import SCALERS

from models.models_TrANN import FONN1, FONN2, FONN3, TREENN1, TREENN2, TREENN3


def main():
    # Load data
    X, y = DataLoader.load_california_data()

    # Create a ModelEvaluator instance
    evaluator = ModelEvaluator()  # added instance creation

    # Base parameters for all models
    base_params = {
        "hidden_dim": MODEL_PARAMS["hidden_dim"],
        "max_iter": MODEL_PARAMS["max_iter"],
        "learning_rate_init": MODEL_PARAMS["learning_rate_init"],
    }

    # Evaluate tree models
    for name, model in [
        ("TREENN1", TREENN1),
        ("TREENN2", TREENN2),
        ("TREENN3", TREENN3),
    ]:
        evaluator.evaluate_model(
            name,
            Pipeline(
                [("scaler", SCALERS["standard"]), ("model", model(**base_params))]
            ),
            X,
            y,
            cv=TRAIN_TEST_SPLIT,
        )

    # Evaluate forest models
    forest_params = {**base_params, "num_trees": MODEL_PARAMS["num_trees"]}
    for name, model in [
        ("FONN1", FONN1),
        ("FONN2", FONN2),
        ("FONN3", FONN3),
    ]:
        evaluator.evaluate_model(
            name,
            Pipeline(
                [("scaler", SCALERS["standard"]), ("model", model(**forest_params))]
            ),
            X,
            y,
            cv=TRAIN_TEST_SPLIT,
        )

    # Evaluate PureMLP
    mlp_params = {
        "hidden_layer_sizes": (MODEL_PARAMS["hidden_dim"],),
        "max_iter": MODEL_PARAMS["max_iter"],
        "learning_rate_init": MODEL_PARAMS["learning_rate_init"],
        "random_state": MODEL_PARAMS["random_state"],
        "n_iter_no_change": MODEL_PARAMS["max_iter"],
    }
    evaluator.evaluate_model(
        "PureMLP",
        Pipeline(
            [("scaler", SCALERS["standard"]), ("model", MLPRegressor(**mlp_params))]
        ),
        X,
        y,
        cv=TRAIN_TEST_SPLIT,
    )

    print(evaluator.results)
    print(evaluator.raw_cv_results)

    output_file = f'Model_results_{MODEL_PARAMS["max_iter"]}_{TRAIN_TEST_SPLIT}_{time.strftime("%F_%T")}.csv'
    evaluator.save_results(f"output/{output_file}")
    evaluator.save_raw_cv_results(f"output/raw_{output_file}")


if __name__ == "__main__":
    main()
