from flask import Flask, request, jsonify
import numpy as np
import keras
from PIL import Image
import io
import os

app = Flask(__name__)

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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({
            "error": "No image file provided"
        }), 400

    file = request.files["file"]

    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({
            "error": "Invalid image file"
        }), 400

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

    return jsonify({
        "predicted_class": class_names[predicted_class],
        "confidence": round(float(confidence), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
