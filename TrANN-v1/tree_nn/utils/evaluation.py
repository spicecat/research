"""Model evaluation utilities."""

import time

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split


class ModelEvaluator:
    """Handle model evaluation and results storage."""

    def __init__(self):
        self.results = []
        self.raw_cv_results = []  # Aggregates raw cv_results

    def evaluate_model(self, name, model_func, X, y, cv=0.2, **kwargs):
        """
        Evaluate a model using cross-validation or train-test split (if cv is a float) and store its results.

        Parameters:
        -----------
        name : str
            Name of the model.
        model_func : class
            sklearn estimator class to evaluate (must implement fit and predict).
        X, y : array-like
            Data used for evaluation.
        cv : int or float
            If an int, use that many folds for cross-validation.
            If a float between 0 and 1, use it as the test size for a train-test split.
        **kwargs : dict
            Additional model parameters.
        """
        if isinstance(cv, float) and 0 < cv < 1:
            self._train_test_split(name, model_func, X, y, cv, **kwargs)
        else:
            self._cross_validation(name, model_func, X, y, cv, **kwargs)

    def _train_test_split(self, name, model_func, X, y, test_size, **kwargs):
        # Split the data into training and test sets using cv as test_size.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        model = model_func(**kwargs)

        # Measure training time.
        start_fit = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_fit

        # Measure scoring time.
        start_score = time.time()
        y_pred = model.predict(X_test)
        score_time = time.time() - start_score

        # Calculate metrics.
        r2_val = r2_score(y_test, y_pred)
        mae_val = mean_absolute_error(y_test, y_pred)
        mse_val = mean_squared_error(y_test, y_pred)

        # Save overall results.
        self.results.append(
            {
                "Model": name,
                "R² Score": r2_val,
                "MAE": mae_val,
                "MSE": mse_val,
                "Time (s)": fit_time + score_time,
            }
        )

        # Save the raw results as a single "fold" (fold 0).
        self.raw_cv_results.append(
            {
                "Model": name,
                "Fold": 0,
                "Test R2": r2_val,
                "Test MAE": mae_val,
                "Test MSE": mse_val,
                "Fit Time": fit_time,
                "Score Time": score_time,
                "kwargs": kwargs,
            }
        )

    def _cross_validation(self, name, model_func, X, y, cv, **kwargs):
        model = model_func(**kwargs)
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
        }
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, return_train_score=False
        )

        self.results.append(
            {
                "Model": name,
                "R² Score": cv_results["test_r2"].mean(),
                "MAE": -cv_results["test_mae"].mean(),
                "MSE": -cv_results["test_mse"].mean(),
                "Time (s)": cv_results["fit_time"].mean()
                + cv_results["score_time"].mean(),
            }
        )

        # Aggregate each fold's raw cv_results into self.raw_cv_results.
        num_folds = len(cv_results["test_r2"])
        for fold in range(num_folds):
            self.raw_cv_results.append(
                {
                    "Model": name,
                    "Fold": fold,
                    "Test R2": cv_results["test_r2"][fold],
                    "Test MAE": -cv_results["test_mae"][fold],
                    "Test MSE": -cv_results["test_mse"][fold],
                    "Fit Time": cv_results["fit_time"][fold],
                    "Score Time": cv_results["score_time"][fold],
                    "kwargs": kwargs,
                }
            )

    def save_results(self, filepath="model_results.csv"):
        """Save evaluation results to CSV."""
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(filepath, index=False)
        return results_df

    def save_raw_cv_results(self, filepath="raw_cv_results.csv"):
        """Save all raw cross-validation results to CSV."""
        raw_df = pd.DataFrame(self.raw_cv_results)
        raw_df.to_csv(filepath, index=False)
        return raw_df
