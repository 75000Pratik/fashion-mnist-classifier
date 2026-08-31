import numpy as np
import matplotlib.pyplot as plt
import keras

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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

probabilities = model.predict(
    X_test,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)

cm = confusion_matrix(
    y_test,
    predictions
)

cm_without_diagonal = cm.copy()

np.fill_diagonal(
    cm_without_diagonal,
    0
)

max_index = np.unravel_index(
    np.argmax(cm_without_diagonal),
    cm_without_diagonal.shape
)

actual_class = max_index[0]
predicted_class = max_index[1]

print(
    "Most confused:",
    class_names[actual_class],
    "->",
    class_names[predicted_class]
)

print(
    "Number of mistakes:",
    cm_without_diagonal[max_index]

)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title("Fashion-MNIST Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.show()
