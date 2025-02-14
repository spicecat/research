"""Main script for running Tree Neural Network experiments."""

from config.settings import MODEL_PARAMS
from data.data_loader import DataLoader
from sklearn.neural_network import MLPRegressor

from utils.evaluation import ModelEvaluator
from models.models_TrANN import TREENN1, TREENN2, TREENN3, FONN1, FONN2, FONN3


def main():
    # Load data
    X, y = DataLoader.load_boston_data()

    # Create a ModelEvaluator instance
    evaluator = ModelEvaluator()  # added instance creation

    # Base parameters for all models
    base_params = {
        "hidden_dim": MODEL_PARAMS["hidden_dim"],
    }

    # Evaluate tree models
    for name, model in [
        ("TREENN1", TREENN1),
        ("TREENN2", TREENN2),
        ("TREENN3", TREENN3),
    ]:
        evaluator.evaluate_model(name, model, X, y, **base_params)

    # Evaluate forest models
    forest_params = {**base_params, "num_trees": MODEL_PARAMS["num_trees"]}
    for name, model in [("FONN1", FONN1), ("FONN2", FONN2), ("FONN3", FONN3)]:
        evaluator.evaluate_model(name, model, X, y, **forest_params)

    # Evaluate PureMLP
    mlp_params = {
        "hidden_layer_sizes": (MODEL_PARAMS["hidden_dim"],),
        "max_iter": MODEL_PARAMS["max_iter"],
        "learning_rate_init": MODEL_PARAMS["learning_rate_init"],
        "random_state": MODEL_PARAMS["random_state"],
        "n_iter_no_change": MODEL_PARAMS["max_iter"],
    }
    evaluator.evaluate_model("PureMLP", MLPRegressor, X, y, **mlp_params)

    evaluator.save_results()
    evaluator.save_raw_cv_results()


if __name__ == "__main__":
    main()
