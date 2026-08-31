import matplotlib.pyplot as plt
import keras
from cnn_data_loader import load_cnn_data

X_train, y_train, X_test, y_test = load_cnn_data()

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),

    keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    keras.layers.Flatten(),

    keras.layers.Dense(
        128,
        activation="relu"
    ),

    keras.layers.Dropout(0.3),

    keras.layers.Dense(
        10,
        activation="softmax"
    )
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "models/best_fashion_mnist_cnn.keras",
    monitor="val_loss",
    save_best_only=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig("outputs/accuracy_plot.png")
plt.show()

plt.figure()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.savefig("outputs/loss_plot.png")
plt.show()

model.save("models/fashion_mnist_cnn.keras")
print("CNN model saved successfully.")
