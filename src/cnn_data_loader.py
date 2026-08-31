import keras 

def load_cnn_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    load_cnn_data()