import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.inference import CLASS_NAMES


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    return fig


def plot_knn_accuracies(k_values, accuracies):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, accuracies, marker='o')
    ax.title('KNN Accuracy vs K Value')
    ax.set_xlabel('K Value')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(k_values)
    ax.grid()
    
    return fig


def plot_prediction_probabilities(prediction_probs):
    if prediction_probs.ndim == 2 and prediction_probs.shape[0] == 1:
        prediction_probs = prediction_probs.flatten()

    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": prediction_probs
    })
    fig, ax = plt.subplots()
    df.plot(kind="bar", x="Class", y="Probability", ax=ax, legend=False)
    ax.set_ylim(0, 1)
    ax.set_title('Prediction Probabilities')
    ax.set_ylabel('Probability')
    
    return fig