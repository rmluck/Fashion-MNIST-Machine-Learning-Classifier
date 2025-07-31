"""
Neural Network Model for classifying Fashion MNIST dataset.
"""

# Import necessary libraries
import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Function to create and train a neural network model
def train_neural_network(x_train, y_train, hidden_layer_size, activation, solver, alpha, batch_size):
    # Initialize the neural network model with specified parameters
    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation=activation,
                          solver=solver, alpha=alpha, batch_size=batch_size, max_iter=1000)

    # Train the model
    model.fit(x_train, y_train)

    # Return the trained model
    return model


# Function to evaluate the neural network model
def evaluate_neural_network(model, x_test, y_test):
    # Predict the labels for the test data
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the evaluation results
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Return the accuracy
    return accuracy


# Function to perform hyperparameter search for neural network
def hyperparameter_search_neural_network(x_train, y_train):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'batch_size': [32, 64, 128]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, cv=3, scoring='accuracy')

    # Fit the grid search to the training data
    grid_search.fit(x_train, y_train)

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Return the best parameters and score
    return best_params, best_score


# Function to scale the input data for neural network
def neural_network_scaling(x_train, x_test):
    # Scale the input data using StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test data
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Return the scaled training and test data
    return x_train_scaled, x_test_scaled