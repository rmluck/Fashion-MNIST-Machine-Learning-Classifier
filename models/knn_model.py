import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def train_knn_model(x_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    return knn


def evaluate_knn_model(knn, x_test, y_test):
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    return accuracy


def find_best_k(x_train, y_train, x_test, y_test):
    best_k = None
    best_accuracy = 0.0
    
    for k in k_values:
        knn = train_knn_model(x_train, y_train, k)
        accuracy = evaluate_knn_model(knn, x_test, y_test)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            
    return best_k, best_accuracy