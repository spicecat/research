"""Model evaluation utilities."""

import time
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ModelEvaluator:
    """Handle model evaluation and results storage."""

    def __init__(self):
        self.results = []

    def evaluate_model(
        self, name, model_func, X_train, X_test, y_train, y_test, **kwargs
    ):
        """
        Evaluate a model and store its results.

        Parameters:
        -----------
        name : str
            Name of the model
        model_func : class
            Model class to evaluate
        X_train, X_test, y_train, y_test : array-like
            Training and test data
        **kwargs : dict
            Additional model parameters
        """
        start_time = time.time()
        model = model_func(**kwargs)
        model.train(X_train, y_train)
        predictions = model.forward(X_test)
        end_time = time.time()

        self.results.append(
            {
                "Model": name,
                "R² Score": r2_score(y_test, predictions),
                "MAE": mean_absolute_error(y_test, predictions),
                "MSE": mean_squared_error(y_test, predictions),
                "Time (s)": end_time - start_time,
            }
        )

    def save_results(self, filepath="model_results.csv"):
        """Save evaluation results to CSV."""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(filepath, index=False)
        return results_df
