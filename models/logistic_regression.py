"""
Logistic Regression Model for for classifying Fashion MNIST dataset.
"""

# Import necessary libraries
import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Function to train a logistic regression model
def train_logistic_regression_model(x_train, y_train):
    # Initialize the logistic regression model
    model = LogisticRegression(max_iter=1000)

    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Return the trained model
    return model


# Function to evaluate the logistic regression model
def evaluate_logistic_regression_model(model, x_test, y_test):
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


# Function to perform hyperparameter search for logistic regression
def hyperparameter_search_logistic_regression(x_train, y_train, x_test, y_test):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the training data
    grid_search.fit(x_train, y_train)
    
    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Return the best parameters and score
    return best_params, best_score