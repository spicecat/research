import keras


def scale_data(X, y):
    # Standardize the features
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    return (X_scaler.fit_transform(X),
            y_scaler.fit_transform(y.reshape(-1, 1)),
            X_scaler,
            y_scaler)


def compile_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(units=64, activation='sigmoid',
                           input_shape=input_shape),
        keras.layers.Dense(units=64, activation='sigmoid'),
        keras.layers.Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mae')

    return model


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import Bunch

    # Load the diabetes dataset
    diabetes: Bunch = load_diabetes()  # type: ignore
    X, y = diabetes.data, diabetes.target

    # Standardize the features
    X, y, X_scaler, y_scaler = scale_data(X, y)

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
    print(y_scaler.inverse_transform(y_pred))
    print(y_scaler.inverse_transform(y))

    loss_df = pd.DataFrame(losses.history)
    loss_df.loc[:, ['loss', 'val_loss']].plot()
    plt.show()
    plt.savefig('output/loss.png')
