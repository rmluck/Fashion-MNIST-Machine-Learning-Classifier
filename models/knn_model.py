"""
K-Nearest Neighbors (KNN) model for classifying Fashion MNIST dataset.
"""

# Import necessary libraries
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Define the range of k values to test
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# Function to create and train a KNN model
def train_knn_model(x_train, y_train, k):
    # Initialize the KNN model with the specified number of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the model to the training data
    knn.fit(x_train, y_train)

    # Return the trained KNN model
    return knn


# Function to evaluate the KNN model
def evaluate_knn_model(knn, x_test, y_test):
    # Predict the labels for the test data
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the evaluation results
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    # Return the accuracy
    return accuracy


# Function to find the best k value based on accuracy
def find_best_k(x_train, y_train, x_test, y_test):
    # Initialize variables to track the best k and its accuracy
    best_k = None
    best_accuracy = 0.0
    
    # Iterate through the k values to find the best one
    for k in k_values:
        # Train the KNN model with the current k value
        knn = train_knn_model(x_train, y_train, k)

        # Evaluate the model and get the accuracy
        accuracy = evaluate_knn_model(knn, x_test, y_test)
        
        # If the current accuracy is better than the best found so far, update best_k and best_accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            
    # Print the best k and its accuracy
    return best_k, best_accuracy