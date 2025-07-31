"""
Provides utility functions for model inference, including preparing input shapes, making predictions, and generating sample images.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from tensorflow.keras.datasets import fashion_mnist
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Define class names for Fashion MNIST dataset
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# Function to prepare the input shape for the model
def prepare_input_shape(data, model_type):
    # Check if the model type is CNN or not
    if model_type == "cnn":
        # Reshape the data for CNN input
        if len(data.shape) == 2 and data.shape == (28, 28):
            return data.reshape(1, 28, 28, 1)
        elif len(data.shape) == 3 and data.shape == (28, 28, 1):
            return data.reshape(1, 28, 28, 1)
        elif len(data.shape) == 4:
            return data
        else:
            raise ValueError("Invalid input shape for CNN model.")
    else:
        # Reshape the data for non-CNN models
        if len(data.shape) == 2 and data.shape == (28, 28):
            return data.reshape(1, 784)
        elif len(data.shape) == 2 and data.shape[1] != 784:
            return data.reshape(data.shape[0], -1)
        elif 3 <= len(data.shape) <= 4:
            return data.reshape(data.shape[0], -1)
        return data


# Function to make predictions using the model
def predict_model(model, image_array, model_type):
    # Prepare the input shape based on the model type
    image = prepare_input_shape(image_array, model_type)
    if model_type == "cnn":
        prediction = model.predict(image)[0]
    else:
        prediction = model.predict_proba(image)[0]
    
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)
    
    # Get the class name
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    
    # Get the confidence score
    confidence = prediction[predicted_class_index]

    # Return the predicted class name, confidence score, and prediction array
    return predicted_class_name, confidence, prediction


# Function to generate sample images from the Fashion MNIST dataset
def generate_sample_images():
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    for i in range(20):
        plt.imsave(f"static/sample_images/sample_test_{i}_{y_test[i]}.png", x_test[i], cmap="gray")