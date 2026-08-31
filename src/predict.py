import numpy as np
import keras
from data_loader import load_data

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
    "Ankle boot"

]

X_train, y_train, X_test, y_test = load_data()

model = keras.models.load_model(
    "models/fashion_mnist_model.keras"
)

sample = X_test[10].reshape(1, 784)

probabilities = model.predict(sample, verbose=0)

predicted_class = np.argmax(probabilities[0])

print("Predicted class:", class_names[predicted_class])
print("Actual class:", class_names[y_test[10]])
print("Confidence:", probabilities[0][predicted_class])
