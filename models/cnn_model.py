import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def train_cnn_model(input_shape):
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
    
    return model

def evaluate_cnn_model(model, x_test, y_test):
    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    return accuracy