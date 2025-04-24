# This script needs these libraries to be installed:
#   numpy, sklearn

import wandb
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# load and process data
wbcd = datasets.load_breast_cancer()
feature_names = wbcd.feature_names
labels = wbcd.target_names

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    wbcd.data, wbcd.target, test_size=test_size
)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
model_params = model.get_params()

# get predictions
y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# start a new wandb run and add your model hyperparameters
run = wandb.init(project="my-awesome-project", config=model_params)

# Add additional configs to wandb
run.config.update(
    {
        "test_size": test_size,
        "train_len": len(X_train),
        "test_len": len(X_test),
    }
)

# log additional visualisations to wandb
plot_class_proportions(y_train, y_test, labels)
plot_learning_curve(model, X_train, y_train)
plot_roc(y_test, y_probas, labels)
plot_precision_recall(y_test, y_probas, labels)
plot_feature_importances(model)

# [optional] finish the wandb run, necessary in notebooks
run.finish()