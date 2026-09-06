import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

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
    "Ankle boot"
]

X_train, y_train, X_test, y_test = load_cnn_data()

model = keras.models.load_model(
    "models/best_fashion_mnist_cnn.keras"
)

image = X_test[0]

input_image = np.expand_dims(
    image,
    axis=0
)

last_conv_layer = None

for layer in model.layers:
    if isinstance(layer, keras.layers.Conv2D):
        last_conv_layer = layer


print(
    "Last convolutional layer:",
    last_conv_layer.name
)

predictions = model.predict(
    input_image,
    verbose=0
)

predicted_class = np.argmax(
    predictions[0]
)


print(
    "Predicted class:",
    class_names[predicted_class]
)

print(
    "Actual class:",
    class_names[y_test[0]]
)

inputs = keras.Input(
    shape=(28, 28, 1)
)

x = inputs

conv_output = None


for layer in model.layers:

    if isinstance(
        layer,
        keras.layers.InputLayer
    ):
        continue

    x = layer(x)

    if layer.name == last_conv_layer.name:
        conv_output = x


grad_model = keras.Model(
    inputs=inputs,
    outputs=[
        conv_output,
        x
    ]
)

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        input_image,
        training=False
    )

    class_score = predictions[
        :,
        predicted_class
    ]


grads = tape.gradient(
    class_score,
    conv_outputs
)

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)

conv_outputs = conv_outputs[0]

heatmap = tf.matmul(
    conv_outputs,
    pooled_grads[..., tf.newaxis]
)

heatmap = tf.squeeze(
    heatmap
)

heatmap = tf.maximum(
    heatmap,
    0
)

heatmap = heatmap / (
    tf.reduce_max(heatmap) + 1e-8
)


heatmap = heatmap.numpy()


print(
    "Heatmap shape:",
    heatmap.shape
)

plt.figure()

plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title(
    f"Original Image - {class_names[predicted_class]}"
)

plt.axis("off")

plt.show()

plt.figure()

plt.imshow(
    heatmap,
    cmap="jet"
)

plt.title(
    "Grad-CAM Heatmap"
)

plt.axis("off")

plt.show()

heatmap_resized = tf.image.resize(
    heatmap[..., np.newaxis],
    (28, 28)
).numpy().squeeze()

plt.figure()

plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.imshow(
    heatmap_resized,
    cmap="jet",
    alpha=0.4
)

plt.title(
    f"Grad-CAM - {class_names[predicted_class]}"
)

plt.axis("off")

plt.savefig(
    "outputs/gradcam.png"
)

plt.show()