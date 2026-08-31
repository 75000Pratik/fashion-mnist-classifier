import numpy as np
import keras
from cnn_data_loader import load_cnn_data

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

X_train, y_train, X_test, y_test = load_cnn_data()

model = keras.models.load_model(
    "models/fashion_mnist_cnn.keras"
)

sample = X_test[0].reshape(1, 28, 28, 1)

probabilities = model.predict(sample, verbose=0)

predicted_class = np.argmax(probabilities[0])

print("Predicted class index:", predicted_class)
print("Number of class names:", len(class_names))

print("Predicted class:", class_names[predicted_class])
print("Actual class:", class_names[y_test[0]])
print("Confidence:", probabilities[0][predicted_class])
