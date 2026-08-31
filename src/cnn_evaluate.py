import keras
from cnn_data_loader import load_cnn_data

X_train, y_train, X_test, y_test = load_cnn_data()

model = keras.models.load_model(
    "models/best_fashion_mnist_cnn.keras"
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("CNN Test loss:", test_loss)
print("Cnn Test accuracy:", test_accuracy)
