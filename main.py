from tree.regression_tree import bootstrap_predictions
from neuralnet.neuralnet import scale_data, compile_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
import random


def set_seed(seed):
    import tensorflow as tf
    # Set the random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


set_seed(42)

# Load the diabetes dataset
diabetes: Bunch = load_diabetes()  # type: ignore
X, y = diabetes.data[:50], diabetes.target[:50]
print("Data shape:", X.shape, y.shape)

# Bootstrap columns
reps = bootstrap_predictions(X, y)
X_extended = np.concatenate((X, reps), axis=1)
print(X_extended)

X, y, X_scaler, y_scaler = scale_data(X_extended, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
input_shape = [X_train.shape[1]]

model = compile_model(input_shape)
model.summary()

losses = model.fit(X_train, y_train,
                   validation_data=(X_test, y_test),
                   batch_size=256,
                   epochs=15,)

y_pred = model.predict(X)
# print(y_scaler.inverse_transform(y_pred))
# print(y_scaler.inverse_transform(y))

loss_df = pd.DataFrame(losses.history)
loss_df.loc[:, ['loss', 'val_loss']].plot()
plt.savefig('output/loss.png')
# plt.show()

# SHAP analysis
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=diabetes.feature_names + ['bootstrap' + str(i) for i in range(reps.shape[1])])
plt.savefig('output/shap_summary.png')

shap.plots.waterfall(shap_values[0])
plt.savefig('output/shap_waterfall.png')
