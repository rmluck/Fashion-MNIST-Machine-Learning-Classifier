import os
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.data import load_data


def train_neural_network(x_train, y_train, hidden_layer_size, activation, solver, alpha, batch_size):
    # Create the MLP model
    model = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation=activation,
                          solver=solver, alpha=alpha, batch_size=batch_size, max_iter=1000)

    # Train the model
    model.fit(x_train, y_train)
    return model


def evaluate_neural_network(model, x_test, y_test):
    # Evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    return accuracy


def hyperparameter_search_neural_network(x_train, y_train):
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'batch_size': [32, 64, 128]
    }
    grid_search = GridSearchCV(MLPClassifier(max_iter=1000), param_grid, cv=3, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    return best_params, best_score


def neural_network_scaling(x_train, x_test):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled