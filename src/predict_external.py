import numpy as np
import keras
from PIL import Image

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

model = keras.models.load_model(
    "models/best_fashion_mnist_cnn.keras"
)

image = Image.open("sample_image.png")

image = image.convert("L")

image = image.resize((28, 28))

image_array = np.array(image)

image_array = 255 - image_array

image_array = image_array / 255.0

image_array = image_array.reshape(1, 28, 28, 1)

probabilities = model.predict(
    image_array,
    verbose=0
)

predicted_class = np.argmax(probabilities[0])


confidence = probabilities[0][predicted_class] * 100

print("Predicted class:", class_names[predicted_class])
print(f"Confidence: {confidence:.2f}%")
