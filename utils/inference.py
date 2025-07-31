import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from tensorflow.keras.datasets import fashion_mnist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def prepare_input_shape(data, model_type):
    if model_type == "cnn":
        if len(data.shape) == 2 and data.shape == (28, 28):
            return data.reshape(1, 28, 28, 1)
        elif len(data.shape) == 3 and data.shape == (28, 28, 1):
            return data.reshape(1, 28, 28, 1)
        elif len(data.shape) == 4:
            return data
        else:
            raise ValueError("Invalid input shape for CNN model.")
    else:
        if len(data.shape) == 2 and data.shape == (28, 28):
            return data.reshape(1, 784)
        elif len(data.shape) == 2 and data.shape[1] != 784:
            return data.reshape(data.shape[0], -1)
        elif 3 <= len(data.shape) <= 4:
            return data.reshape(data.shape[0], -1)
        return data

def predict_model(model, image_array, model_type):
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

    return predicted_class_name, confidence, prediction


def generate_sample_image():
    (x_train, y_train), _ = fashion_mnist.load_data()
    plt.imsave("static/sample_images/sample_image.png", x_train[0], cmap='gray')