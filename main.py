import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models_TrANN import FONN1, FONN2, FONN3, TREENN1, TREENN2, TREENN3

one_hot_encoder = OneHotEncoder(sparse_output=False)

# Load the Boston dataset
dataset = "boston"
raw_df = pd.read_csv("data/boston.csv")
X = raw_df.drop(columns=["MEDV"]).values
y = raw_df["MEDV"].values

# # Load LengthOfStay
# dataset = "LengthOfStay"
# raw_df = pd.read_csv("data/LengthOfStay.csv")
# categorical_cols = raw_df.select_dtypes(
#     include=["object", "category"]).columns.tolist()
# # raw_df = pd.get_dummies(raw_df, columns=categorical_cols, drop_first=True)
# raw_df = raw_df.drop(columns=categorical_cols)
# X = raw_df.drop(columns=["eid", "lengthofstay"]).values
# y = raw_df["lengthofstay"].values

# Load HospitalStay
# dataset = "HospitalStay"
# raw_df = pd.read_csv("data/Healthcare_Investments_and_Hospital_Stay.csv")
# categorical_cols = raw_df.select_dtypes(
#     include=["object", "category"]).columns.tolist()
# # one_hot_encoded = one_hot_encoder.fit_transform(raw_df[categorical_cols])
# # raw_df = pd.concat([raw_df.drop(columns=categorical_cols),
# #                     pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out())], axis=1)
# raw_df = raw_df.drop(columns=categorical_cols)
# X = raw_df.drop(columns=["Hospital_Stay"]).values
# y = raw_df["Hospital_Stay"].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Results storage
results = []


def evaluate_model(name, model, n_folds=3):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_predictions = []
    all_true_values = []

    start_time = time.time()

    for train_idx, val_idx in kf.split(X_train):
        # Split data for this fold
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        # Train model on this fold
        model.fit(X_fold_train, y_fold_train)

        # Get predictions
        predictions = model.predict(X_fold_val)

        # Store predictions and true values
        all_predictions.extend(predictions)
        all_true_values.extend(y_fold_val)

    end_time = time.time()
    comp_time = end_time - start_time

    # TODO: use mean of metrics
    # Calculate metrics on aggregated predictions
    results.append(
        {
            "Model": name,
            "R² Score": r2_score(all_true_values, all_predictions),
            "MAE": mean_absolute_error(all_true_values, all_predictions),
            "MSE": mean_squared_error(all_true_values, all_predictions),
            "Time (s)": comp_time,
        }
    )


input_dim = X_train.shape[1]
hidden_dim = 20
output_dim = 1
max_iter = 400
learning_rate_init = 0.02

treenn1 = TREENN1(input_dim, hidden_dim, output_dim, learning_rate_init, max_iter)
treenn2 = TREENN2(input_dim, hidden_dim, output_dim, learning_rate_init, max_iter)
treenn3 = TREENN3(input_dim, hidden_dim, output_dim, learning_rate_init, max_iter)
fonn1 = FONN1(input_dim, hidden_dim, output_dim, 20, learning_rate_init, max_iter)
fonn2 = FONN2(input_dim, hidden_dim, output_dim, 20, learning_rate_init, max_iter)
fonn3 = FONN3(input_dim, hidden_dim, output_dim, 20, learning_rate_init, max_iter)


# Evaluate all models
evaluate_model("TREENN1", treenn1)
evaluate_model("TREENN2", treenn2)
evaluate_model("TREENN3", treenn3)
evaluate_model("FONN1", fonn1)
evaluate_model("FONN2", fonn2)
evaluate_model("FONN3", fonn3)

# Evaluate PureMLP
start_time = time.time()
mlp = MLPRegressor(
    hidden_layer_sizes=(20,),
    max_iter=max_iter,
    learning_rate_init=learning_rate_init,
    learning_rate="adaptive",
    n_iter_no_change=100000,
    random_state=42,
)
mlp.fit(X_train, y_train)
print(mlp.n_iter_)
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
output_file = f"output/results_{dataset}_{max_iter}_{str(learning_rate_init)[2:]}_{time.strftime('%F_%T')}"
results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
print(results_df)

# other dataset
# time limit
# hospital stay length data
# categorical

# conda activate research && make && clear && python3 main.pyc
