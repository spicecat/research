import os
import time
import warnings

import numpy as np

# Optuna for HPO
import optuna
import pandas as pd
import tensorflow as tf
from optuna.integration import OptunaSearchCV

# Use the modern scikeras wrapper
from scikeras.wrappers import KerasRegressor

# Scikit-learn imports
from sklearn import linear_model, preprocessing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from tensorflow import keras
from keras.layers import Dense, Dropout, Input  # type: ignore
from keras.models import Sequential  # type: ignore
from keras.optimizers import Adam  # type: ignore

print(f"TensorFlow version: {tf.__version__}")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress verbose TF logging (0 = all messages are logged (default), 1 = INFO messages are filtered out, 2 = WARNING messages are filtered out, 3 = ERROR messages are filtered out)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")  # Suppress TensorFlow INFO/WARNING messages


################################################################################
# Configuration
################################################################################

# Data file
DATA_FILE = "ML_sample.dta"

# Characteristics
CharsVars = [
    "absacc",
    "beta",
    "cfp",
    "chpmia",
    "ep",
    "gma",
    "herf",
    "idiovol",
    "lev",
    "mom12m",
    "mom6m",
    "nincr",
    "pchdepr",
    "ps",
    "roavol",
    "roeq",
    "salecash",
    "stdcf",
    "sue",
    "tang",
]

# Sample split definitions
# Tuning sample (Train+Validation): 2000 ~ 2014 (15 years)
# Testing sample                : 2015 ~ 2019 (5 years)
ym_tune_st = (2000 - 1960) * 12
ym_tune_ed = (2014 - 1960) * 12 + 11
ym_test_st = (2015 - 1960) * 12
ym_test_ed = (2019 - 1960) * 12 + 11

# Model Selection & Tuning Parameters
navg = 5  # Number of top models to average for ensemble
N_SPLITS_CV = 5  # Number of folds for TimeSeriesSplit
N_TRIALS_OPTUNA = 30  # Number of Optuna trials (adjust per model complexity/time)
RANDOM_STATE = 42  # For reproducibility

################################################################################
# Helper Functions
################################################################################


def winsorize(X, q=0.01):
    """Winsorizes a pandas Series or numpy array."""
    X = X.copy()
    # Use nanquantile to handle potential NaNs gracefully
    q_l = np.nanquantile(X, q)
    q_u = np.nanquantile(X, 1 - q)
    # Check if quantiles are NaN (can happen if all data is NaN or q is extreme)
    if pd.isna(q_l) or pd.isna(q_u):
        return X  # Return original if quantiles are invalid
    X[X < q_l] = q_l
    X[X > q_u] = q_u
    return X


def cal_r2(y_true, y_pred):
    """
    Calculates R-squared following Gu, Kelly, and Xiu (2020).
    Denominator is sum of squared true values (no demeaning).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Handle potential NaNs if any slipped through
    valid_idx = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]
    if len(y_true) == 0:
        return np.nan

    ss_res = np.sum(np.square(y_true - y_pred))
    ss_tot_no_demean = np.sum(np.square(y_true))

    if ss_tot_no_demean == 0:
        return 1.0 if ss_res == 0 else -np.inf  # Avoid division by zero

    return 100 * (1 - ss_res / ss_tot_no_demean)


# Create a scorer object for use with OptunaSearchCV/sklearn CV functions
r2_scorer = make_scorer(cal_r2, greater_is_better=True)

################################################################################
# Data Loading and Preprocessing
################################################################################
start_time = time.time()
print(f"--- Data Loading and Preprocessing ---")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Data file '{DATA_FILE}' not found. Please ensure it's in the correct directory."
    )

print(f"Loading data from {DATA_FILE}...")
retdata = pd.read_stata(DATA_FILE)
retdata["ym"] = (retdata["year"] - 1960) * 12 + (retdata["month"] - 1)
retdata = retdata.astype({"permno": "int", "year": "int", "month": "int", "ym": "int"})
# Select necessary columns and sort
retdata = retdata[
    ["permno", "year", "month", "ym", "lme", "retadj", "exret"] + CharsVars
]
retdata = retdata.sort_values(by=["ym", "permno"], ascending=True).reset_index(
    drop=True
)
assert not retdata.duplicated(
    subset=["ym", "permno"]
).any(), "Duplicate permno-ym entries found!"
print(f"Data loaded. Shape: {retdata.shape}")

# Winsorize excess returns for the tuning sample period only
print("Winsorizing excess returns (q=0.01) for tuning period...")
# Identify rows belonging to the tuning period for winsorization
tune_period_mask = (retdata["ym"] >= ym_tune_st) & (retdata["ym"] <= ym_tune_ed)
retdata["exret_winsor"] = retdata["exret"]  # Initialize column
retdata.loc[tune_period_mask, "exret_winsor"] = (
    retdata.loc[tune_period_mask]
    .groupby("ym")["exret"]
    .transform(lambda x: winsorize(x, q=0.01))
)
# For periods outside tuning, keep original exret in exret_winsor (or set to NaN if preferred)
# Here, we keep original, as Y_tune will be sliced later.

# Time index dictionary (for standardization)
ym_dc = {}
for i, ym_i in enumerate(retdata["ym"]):
    ym_dc.setdefault(ym_i, []).append(i)
ym_ls = sorted(ym_dc.keys())

# Cross-sectional standardization (using numpy for speed)
print("Performing cross-sectional standardization on characteristics...")
CMat = retdata[CharsVars].values.astype(np.float64)  # Ensure float64 for precision

for ym_i in ym_ls:
    idx_i = ym_dc[ym_i]
    if len(idx_i) > 1:
        month_data = CMat[idx_i, :]
        # Use nanmean/nanstd for robustness if NaNs exist before scaling
        mean = np.nanmean(month_data, axis=0)
        std = np.nanstd(month_data, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero for constant columns
        CMat[idx_i, :] = (month_data - mean) / std
    elif len(idx_i) == 1:
        # If only one sample, set standardized value to 0 (mean) after handling NaNs
        is_nan_row = np.isnan(CMat[idx_i, :])
        CMat[idx_i, ~is_nan_row] = 0

# Fill remaining NaNs (original NaNs or those from single-sample months) with 0
CMat[np.isnan(CMat)] = 0
retdata[CharsVars] = CMat
del CMat, ym_dc, ym_ls  # Clean up memory
print("Standardization complete.")

################################################################################
# Define Tuning and Test Sets
################################################################################

print("Splitting data into tuning and test sets...")
# Find the actual row indices corresponding to the time periods
try:
    idx_tune_st = retdata[retdata["ym"] >= ym_tune_st].index.min()
    idx_tune_ed = retdata[retdata["ym"] <= ym_tune_ed].index.max()
    idx_test_st = retdata[retdata["ym"] >= ym_test_st].index.min()
    idx_test_ed = retdata[retdata["ym"] <= ym_test_ed].index.max()

    # Handle cases where exact start/end months might be missing data
    if pd.isna(idx_tune_st):
        idx_tune_st = retdata.index.min()  # Should not happen if ym_tune_st exists
    if pd.isna(idx_tune_ed):
        idx_tune_ed = retdata[retdata["ym"] < ym_test_st].index.max()
    if pd.isna(idx_test_st):
        idx_test_st = retdata[retdata["ym"] > ym_tune_ed].index.min()
    if pd.isna(idx_test_ed):
        idx_test_ed = retdata.index.max()  # Should not happen if ym_test_ed exists

    print(f"Tuning indices: {idx_tune_st} to {idx_tune_ed}")
    print(f"Test indices: {idx_test_st} to {idx_test_ed}")

    # Tuning sample (2000-2014) -> Use winsorized returns for Y
    X_tune = retdata.loc[idx_tune_st:idx_tune_ed, CharsVars].values
    Y_tune = retdata.loc[idx_tune_st:idx_tune_ed, "exret_winsor"].values

    # Testing sample (2015-2019) -> Use original excess returns for Y
    X_test = retdata.loc[idx_test_st:idx_test_ed, CharsVars].values
    Y_test = retdata.loc[idx_test_st:idx_test_ed, "exret"].values

    # Check for potential issues
    if X_tune.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Tuning or Test set is empty. Check date ranges and data.")
    if np.isnan(X_tune).any() or np.isnan(Y_tune).any():
        print(
            "Warning: NaNs found in tuning data after preprocessing. This might indicate an issue."
        )
    if np.isnan(X_test).any() or np.isnan(Y_test).any():
        print("Warning: NaNs found in test data. This might indicate an issue.")

    print(f"X_tune shape: {X_tune.shape}, Y_tune shape: {Y_tune.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

except Exception as e:
    print(f"Error determining indices or creating data splits: {e}")
    raise

# Clean up original dataframe to save memory
del retdata
print(f"Data preparation took {time.time() - start_time:.2f} seconds.")

################################################################################
# TimeSeriesSplit Cross-Validator
################################################################################
tscv = TimeSeriesSplit(
    n_splits=N_SPLITS_CV, gap=0
)  # gap=0 assumes predicting next period

################################################################################
# Model Training and Hyperparameter Tuning with OptunaSearchCV
################################################################################

tuned_results = {}
model_times = {}

# --- 1. Linear Regression (No Hyperparameter Tuning Needed) ---
model_name = "LinearRegression"
print(f"\n--- Training {model_name} ---")
start_model_time = time.time()
lr_model = linear_model.LinearRegression()
try:
    lr_model.fit(X_tune, Y_tune)
    # Evaluate on the test set directly
    Y_pred_lr_test = lr_model.predict(X_test)
    lr_test_r2 = cal_r2(Y_test, Y_pred_lr_test)
    print(f"{model_name} Test R2 (using full tuning set fit): {lr_test_r2:.4f}")
    # Use test R2 as its 'score' for ranking purposes.
    tuned_results[model_name] = {"best_estimator": lr_model, "best_score": lr_test_r2}
except Exception as e:
    print(f"Error training/evaluating {model_name}: {e}")
    tuned_results[model_name] = {
        "best_estimator": None,
        "best_score": -np.inf,
    }  # Mark as failed
model_times[model_name] = time.time() - start_model_time
print(f"{model_name} training & evaluation took {model_times[model_name]:.2f}s.")


# --- 2. Gradient Boosting Regressor (GBRT) ---
model_name = "GBRT"
print(f"\n--- Tuning {model_name} with Optuna ({N_TRIALS_OPTUNA} trials) ---")
start_model_time = time.time()
gbrt = GradientBoostingRegressor(random_state=RANDOM_STATE, loss="squared_error")

gbrt_param_distributions = {
    "n_estimators": optuna.distributions.IntDistribution(50, 300),
    "learning_rate": optuna.distributions.FloatDistribution(
        0.01, 0.2, log=True
    ),  # Slightly reduced upper bound
    "max_depth": optuna.distributions.IntDistribution(2, 6),  # Common range for GBRT
    "subsample": optuna.distributions.FloatDistribution(
        0.6, 0.9
    ),  # Avoid 1.0 for regularization
    "min_samples_leaf": optuna.distributions.IntDistribution(
        20, 100
    ),  # Increased min leaf size
    "max_features": optuna.distributions.FloatDistribution(
        0.5, 1.0
    ),  # Added max_features
}

gbrt_search = OptunaSearchCV(
    estimator=gbrt,
    param_distributions=gbrt_param_distributions,
    n_trials=N_TRIALS_OPTUNA,
    cv=tscv,
    scoring=r2_scorer,
    n_jobs=-1,  # Use all available CPU cores
    random_state=RANDOM_STATE,
    refit=True,
    verbose=0,  # Set to 1 for progress
)
try:
    gbrt_search.fit(X_tune, Y_tune)
    print(f"Best {model_name} Params: {gbrt_search.best_params_}")
    print(f"Best {model_name} CV R2 Score: {gbrt_search.best_score_:.4f}")
    tuned_results[model_name] = {
        "best_estimator": gbrt_search.best_estimator_,
        "best_score": gbrt_search.best_score_,
    }
except Exception as e:
    print(f"Error during {model_name} tuning: {e}")
    tuned_results[model_name] = {"best_estimator": None, "best_score": -np.inf}
model_times[model_name] = time.time() - start_model_time
print(f"{model_name} tuning took {model_times[model_name]:.2f}s.")


# --- 3. Random Forest Regressor (RF) ---
model_name = "RF"
print(f"\n--- Tuning {model_name} with Optuna ({N_TRIALS_OPTUNA} trials) ---")
start_model_time = time.time()
rf = RandomForestRegressor(
    random_state=RANDOM_STATE, n_jobs=-1
)  # Use n_jobs for RF too

rf_param_distributions = {
    "n_estimators": optuna.distributions.IntDistribution(50, 300),
    "max_depth": optuna.distributions.IntDistribution(5, 25),  # Can be deeper than GBRT
    "min_samples_split": optuna.distributions.IntDistribution(20, 150),  # Wider range
    "min_samples_leaf": optuna.distributions.IntDistribution(10, 80),  # Wider range
    "max_features": optuna.distributions.FloatDistribution(
        0.3, 1.0
    ),  # Fraction of features
}

rf_search = OptunaSearchCV(
    estimator=rf,
    param_distributions=rf_param_distributions,
    n_trials=N_TRIALS_OPTUNA,
    cv=tscv,
    scoring=r2_scorer,
    n_jobs=-1,  # Use all available CPU cores
    random_state=RANDOM_STATE,
    refit=True,
    verbose=0,
)
try:
    rf_search.fit(X_tune, Y_tune)
    print(f"Best {model_name} Params: {rf_search.best_params_}")
    print(f"Best {model_name} CV R2 Score: {rf_search.best_score_:.4f}")
    tuned_results[model_name] = {
        "best_estimator": rf_search.best_estimator_,
        "best_score": rf_search.best_score_,
    }
except Exception as e:
    print(f"Error during {model_name} tuning: {e}")
    tuned_results[model_name] = {"best_estimator": None, "best_score": -np.inf}
model_times[model_name] = time.time() - start_model_time
print(f"{model_name} tuning took {model_times[model_name]:.2f}s.")


# --- 4. Neural Network (NN) using SciKeras ---
model_name = "NN"
print(f"\n--- Tuning {model_name} with Optuna ({N_TRIALS_OPTUNA} trials) ---")
start_model_time = time.time()


# Define the function to build the Keras model - SciKeras style
# It accepts hyperparameters directly.
def build_nn_model(meta, hidden_layer_sizes=(32,), dropout=0.1, lr=0.001):
    n_features_in_ = meta["n_features_in_"]  # Get input dim from metadata
    model = Sequential(name="NN_Regressor")
    model.add(Input(shape=(n_features_in_,), name="Input_Layer"))
    for i, nodes in enumerate(hidden_layer_sizes):
        model.add(Dense(nodes, activation="relu", name=f"Hidden_Layer_{i+1}"))
        model.add(Dropout(dropout, name=f"Dropout_{i+1}"))
    model.add(
        Dense(1, activation="linear", name="Output_Layer")
    )  # Linear activation for regression

    optimizer = Adam(learning_rate=lr)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model


# Instantiate the KerasRegressor wrapper from SciKeras
# Pass tunable training params (epochs, batch_size) directly to the wrapper
# Pass model architecture params via param_distributions prefixed with 'model__'
nn_estimator = KerasRegressor(
    model=build_nn_model,  # Pass the build function
    loss="mean_squared_error",  # Can specify loss here too
    optimizer="adam",  # Can specify optimizer name
    optimizer__learning_rate=0.001,  # Default LR, will be tuned
    model__dropout=0.1,  # Default dropout, will be tuned
    model__hidden_layer_sizes=(32,),  # Default layers, will be tuned
    verbose=0,  # Suppress Keras training logs during CV
)

# Define search space for NN (including training params)
# Use 'model__' prefix for params passed to build_nn_model
nn_param_distributions = {
    "model__hidden_layer_sizes": optuna.distributions.CategoricalDistribution(
        [(32,), (64,), (32, 16), (64, 32)]
    ),  # Example layer structures
    "model__dropout": optuna.distributions.FloatDistribution(0.0, 0.5, step=0.1),
    "optimizer__learning_rate": optuna.distributions.FloatDistribution(
        1e-4, 1e-2, log=True
    ),
    "batch_size": optuna.distributions.CategoricalDistribution([1024, 2048, 4096]),
    "epochs": optuna.distributions.CategoricalDistribution(
        [20, 30, 50]
    ),  # Adjust range as needed
}

# Add callbacks if desired (e.g., EarlyStopping)
# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=0, restore_best_weights=True)
# fit_params = {'callbacks': [early_stopping]}
# Note: When tuning epochs, early stopping might interfere. Often simpler to tune epochs directly.

nn_search = OptunaSearchCV(
    estimator=nn_estimator,
    param_distributions=nn_param_distributions,
    n_trials=N_TRIALS_OPTUNA,  # Use a smaller number for NN initially if slow
    cv=tscv,
    scoring=r2_scorer,
    n_jobs=1,  # IMPORTANT: Start with 1 for Keras/SciKeras due to potential pickling issues
    random_state=RANDOM_STATE,
    refit=True,
    verbose=0,  # OptunaSearchCV verbosity
)

try:
    # Pass fit_params if using callbacks: fit(X_tune, Y_tune, **fit_params)
    nn_search.fit(X_tune, Y_tune)

    print(f"Best {model_name} Params: {nn_search.best_params_}")
    print(f"Best {model_name} CV R2 Score: {nn_search.best_score_:.4f}")
    tuned_results[model_name] = {
        "best_estimator": nn_search.best_estimator_,
        "best_score": nn_search.best_score_,
    }

except Exception as e:
    print(f"\nError during {model_name} tuning: {e}")
    print(f"Skipping {model_name} model for final selection.")
    # Optionally log the full traceback
    # import traceback
    # traceback.print_exc()
    tuned_results[model_name] = {
        "best_estimator": None,
        "best_score": -np.inf,
    }  # Mark as failed

model_times[model_name] = time.time() - start_model_time
print(f"{model_name} tuning took {model_times[model_name]:.2f}s.")

# Model Selection and Final Evaluation
################################################################################
print("\n--- Model Selection and Final Evaluation ---")
total_tuning_time = sum(model_times.values())
print(f"Total time spent on model training/tuning: {total_tuning_time:.2f} seconds.")

# Filter out models that failed or were skipped
valid_results = {
    k: v
    for k, v in tuned_results.items()
    if v["best_estimator"] is not None
    and not np.isnan(v["best_score"])
    and v["best_score"] != -np.inf
}

if not valid_results:
    print("\nERROR: No models were successfully trained or tuned. Exiting.")
    exit()

# Sort models by their cross-validated R2 score (descending)
# Use the 'best_score' which comes from the CV process for tuned models,
# and the placeholder score (test R2) for LR.
sorted_models = sorted(
    valid_results.items(), key=lambda item: item[1]["best_score"], reverse=True
)

print("\nModel Ranking based on CV/Performance Score:")
for i, (name, result) in enumerate(sorted_models):
    score = result["best_score"]
    print(f"{i+1}. {name}: Score = {score:.4f}")

# Select the top 'navg' models
top_n_models = sorted_models[:navg]

if not top_n_models:
    print(f"\nWarning: No models available after filtering, cannot create ensemble.")
else:
    print(
        f"\nSelecting top {len(top_n_models)} models for ensemble prediction on the test set."
    )

    # Generate predictions on the test set for the selected models
    test_predictions = {}
    individual_test_r2s = {}

    for name, result in top_n_models:
        model = result["best_estimator"]
        print(f"   Predicting with {name}...")
        try:
            start_pred_time = time.time()
            Y_pred_test_single = model.predict(X_test)
            # Ensure prediction is a flat numpy array
            Y_pred_test_single = np.asarray(Y_pred_test_single).flatten()

            test_predictions[name] = Y_pred_test_single
            individual_test_r2s[name] = cal_r2(Y_test, Y_pred_test_single)
            pred_time = time.time() - start_pred_time
            print(
                f"   - {name} Test R2: {individual_test_r2s[name]:.4f} (prediction took {pred_time:.2f}s)"
            )

        except Exception as e:
            print(f"   Error predicting with model {name}: {e}")
            # Remove failed model from ensemble consideration
            if name in test_predictions:
                del test_predictions[name]

    # Average the predictions from the successfully predicting selected models
    if len(test_predictions) > 0:
        # Create a DataFrame for easy averaging
        pred_df = pd.DataFrame(test_predictions)
        ensemble_prediction = pred_df.mean(axis=1).values

        # Calculate the final R2 score for the ensemble prediction on the test set
        ensemble_test_r2 = cal_r2(Y_test, ensemble_prediction)
        print(
            f"\nEnsemble (Average of Top {len(test_predictions)}) Test R2: {ensemble_test_r2:.4f}"
        )
    elif len(top_n_models) > 0:
        print(
            "\nNo successful predictions generated from the selected models for the ensemble."
        )
    else:
        # This case should be caught earlier, but as a safeguard:
        print("\nNo models were selected for the ensemble.")


print("\nScript finished.")
