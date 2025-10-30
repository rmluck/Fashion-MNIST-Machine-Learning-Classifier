"""
Runs web application for the project.
"""

# Import necessary libraries
import streamlit as st
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import load_existing_model
from utils.inference import predict_model, prepare_input_shape
from output.visualization import plot_confusion_matrix, plot_prediction_probabilities
from data.data import load_data


st.set_page_config(
    page_title="Fashion MNIST Machine Learning Classifier",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Set the title of the app
st.title("Fashion MNIST ML Classifier")

# Prompt user to select a model
model_choice = st.sidebar.selectbox("Choose a model", ("Logistic Regression", "KNN", "Neural Network", "CNN"))
model, model_type = load_existing_model(model_choice)


# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert image to grayscale and resize to 28x28
    image = image.convert("L")
    image = image.resize((28, 28))

    # Return the image as a numpy array
    return np.array(image).astype('float32') / 255.0


# Function to get prediction probabilities
def get_prediction_probs(model, data, model_type):
    # Prepare the input shape based on the model type
    data = prepare_input_shape(data, model_type)

    # Use the model to predict probabilities
    if hasattr(model, "predict_proba"):
        return model.predict_proba(data)
    else:
        return model.predict(data)


# Function to display the uploaded image and prediction
def display_uploaded_image(file, model, model_type):
    col1, col2 = st.columns(2)
    image = Image.open(file)
    image_array = preprocess_image(image)

    with col1:
        st.image(image, caption="Uploaded Image")
        predicted_class, confidence, _ = predict_model(model, image_array, model_type)
        st.success(f"Prediction: **{predicted_class}** ({confidence:.2%} confidence)")

    with col2:
        fig, ax = plt.subplots()
        ax.imshow(image_array, cmap="gray")
        ax.set_title("Grayscale Input (28x28)")
        st.pyplot(fig)

        prediction_probs = get_prediction_probs(model, image_array, model_type)
        st.pyplot(plot_prediction_probabilities(prediction_probs[0]))


# Function to evaluate the model on the test set
def evaluate_model(model, model_type):
    (x_train, y_train), (x_test, y_test) = load_data()
    y_test_labels = np.argmax(y_test, axis=1)

    y_pred_probs = get_prediction_probs(model, x_test, model_type)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Display evaluation metrics
    st.subheader("Confusion Matrix")
    st.pyplot(plot_confusion_matrix(y_test_labels, y_pred))


# Prompt user to upload images
uploaded_files = st.file_uploader("Upload images of clothing", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
for file in uploaded_files:
    display_uploaded_image(file, model, model_type)

# Display evaluation metrics if checkbox is selected
with st.expander("Show Evaluation Metrics"):
    if st.checkbox("Evaluate model on test set?"):
        evaluate_model(model, model_type)