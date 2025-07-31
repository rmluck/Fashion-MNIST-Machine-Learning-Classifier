"""
Convolutional Neural Network (CNN) model for classifying Fashion MNIST dataset.
"""

# Import necessary libraries
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Function to create and train a CNN model
def train_cnn_model(input_shape):
    # Initialize the CNN model
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    
    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Third convolutional layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Flatten the output
    model.add(Flatten())
    
    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    
    # Output layer
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes for classification
    
    # Return the model
    return model


# Function to evaluate the CNN model
def evaluate_cnn_model(model, x_test, y_test):
    # Predict the labels for the test data
    y_pred = model.predict(x_test)
    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    report = classification_report(y_test, y_pred)
    
    # Print the evaluation results
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    # Return the accuracy
    return accuracy