import keras


def load_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("Minimum pixel value:", X_train.min())
    print("Maximum pixel value:", X_train.max())

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    load_data()
