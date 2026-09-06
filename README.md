# Fashion-MNIST Image Classifier

A deep learning image classification project built with TensorFlow and Keras.

The project classifies Fashion-MNIST clothing images into 10 categories using both a Dense Neural Network and a Convolutional Neural Network (CNN).

## Project Goal

The goal of this project is to build and evaluate neural network models for multi-class image classification, compare a Dense Neural Network with a CNN, and extend the project with model evaluation, Grad-CAM explainability, external-image inference, and a Flask prediction API.

## Key Results

- **CNN Test Accuracy:** 91.28%
- **Dense Neural Network Test Accuracy:** 87.94%
- **10 Fashion-MNIST Classes**
- Confusion matrix and classification report
- Grad-CAM explainability
- External-image inference
- Flask REST API for model serving

## Tech Stack

`Python` `TensorFlow` `Keras` `NumPy` `Matplotlib` `Scikit-learn` `Pillow` `Flask`

## Dataset

Fashion-MNIST contains:

- 60,000 training images
- 10,000 test images
- 28 × 28 grayscale images
- 10 clothing categories

## Classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Models

### Dense Neural Network

Architecture:

- Input: 784 features
- Dense: 128 neurons with ReLU
- Output: 10 neurons with Softmax

**Test Accuracy:** 87.94%

### Convolutional Neural Network

Architecture:

- Conv2D: 32 filters, 3 × 3 kernel, ReLU
- MaxPooling2D: 2 × 2
- Flatten
- Dense: 128 neurons with ReLU
- Dropout: 30%
- Output: 10 neurons with Softmax

**CNN Test Accuracy:** 91.28%

## Techniques Used

- Image normalization
- Dense Neural Networks
- Convolutional Neural Networks
- ReLU activation
- Softmax activation
- Sparse Categorical Cross-Entropy
- Adam optimizer
- Dropout
- Early Stopping
- Model Checkpointing
- Confusion Matrix
- Precision, Recall and F1-score
- Grad-CAM explainability
- External image inference
- Flask API

## Project Structure

```text
fashion-mnist-classifier/
│
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── cnn_data_loader.py
│   ├── cnn_train.py
│   ├── cnn_evaluate.py
│   ├── cnn_predict.py
│   ├── confusion_matrix.py
│   ├── gradcam.py
│   ├── predict_external.py
│   └── app.py
│
├── models/
│   ├── fashion_mnist_model.keras
│   ├── fashion_mnist_cnn.keras
│   └── best_fashion_mnist_cnn.keras
│
├── outputs/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   └── gradcam.png
│
├── requirements.txt
├── .gitignore
└── README.md