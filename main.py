from tree.regression_tree import bootstrap_predictions
from neuralnet.neuralnet import scale_data, compile_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

# Load the diabetes dataset
diabetes: Bunch = load_diabetes()  # type: ignore
X, y = diabetes.data, diabetes.target

# Test bootstrap
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
plt.show()