"""
Loads the Fashion MNIST dataset and preprocesses it for training.
"""

# Import necessary libraries
import os
import sys
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Load and preprocess the Fashion MNIST dataset
def load_data():
    # Load the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Normalize the images to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Reshape the data to include a channel dimension
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Return the preprocessed training and test data
    return (x_train, y_train), (x_test, y_test)