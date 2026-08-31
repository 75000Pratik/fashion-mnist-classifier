import keras
from data_loader import load_data

X_train, y_train, X_test, y_test = load_data()

model = keras.models.load_model(
    "models/fashion_mnist_model.keras"
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
