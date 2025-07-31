"""
Loads, trains, and saves various machine learning models for the Fashion-MNIST dataset.
"""

# Import necessary libraries
import os
import sys
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data import load_data
from models.cnn_model import train_cnn_model
from models.neural_network import train_neural_network, neural_network_scaling
from models.logistic_regression import train_logistic_regression_model
from models.knn_model import train_knn_model, find_best_k


# Function to load an existing model based on user choice
def load_existing_model(model_choice):
    if model_choice == "CNN":
        return load_model("saved_models/cnn_model.h5"), "cnn"
    elif model_choice == "Neural Network":
        return joblib.load("saved_models/neural_network_model.pkl"), "sklearn"
    elif model_choice == "Logistic Regression":
        return joblib.load("saved_models/logistic_regression_model.pkl"), "sklearn"
    elif model_choice == "KNN":
        return joblib.load("saved_models/knn_model.pkl"), "sklearn"
    else:
        raise ValueError("Invalid model choice. Please select a valid model.")


# Function to train and save all models
def train_and_save_models():
    # Load and preprocess the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    # Train and save the CNN model
    print("Training CNN model...")
    x_train_cnn = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test_cnn = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train_cat = to_categorical(y_train, num_classes=10)
    y_test_cat = to_categorical(y_test, num_classes=10)
    cnn_model = train_cnn_model(input_shape=(28, 28, 1))
    cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    cnn_model.fit(x_train_cnn, y_train_cat, epochs=5, batch_size=64, validation_split=0.1)
    cnn_model.save("saved_models/cnn_model.h5")
    print("CNN model saved.")

    # If y_train is one-hot encoded, convert it back to class labels for logistic regression and KNN
    if y_train.ndim == 2 and y_train.shape[1] == 10:
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

    # Train and save the logistic regression model
    print("Training logistic regression model...")
    log_reg_model = train_logistic_regression_model(x_train_flat, y_train)
    joblib.dump(log_reg_model, "saved_models/logistic_regression_model.pkl")
    print("Logistic regression model saved.")

    # Train and save the neural network model
    print("Training neural network model...")
    x_train_scaled, x_test_scaled = neural_network_scaling(x_train_flat, x_test_flat)
    nn_model = train_neural_network(x_train_scaled, y_train, hidden_layer_size=100, activation="relu", solver="adam", alpha=0.0001, batch_size=64)
    joblib.dump(nn_model, "saved_models/neural_network_model.pkl")
    print("Neural network model saved.")

    # Train and save the KNN model
    print("Training KNN model...")
    best_k, _ = find_best_k(x_train_flat, y_train, x_test_flat, y_test)
    knn_model = train_knn_model(x_train_flat, y_train, best_k)
    joblib.dump(knn_model, "saved_models/knn_model.pkl")
    print(f"KNN model with k={best_k} saved.")


if __name__ == "__main__":
    train_and_save_models()