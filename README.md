# Fashion-MNIST-Classifier

*By Rohan Mistry - Last updated July 31, 2025*

---

## 📖 Overview

Web application developed using Python and TensorFlow/Keras to classify images of clothing from the Fashion MNIST dataset. Users can upload grayscale 28x28 images of apparel items to receive predictions from several machine learning models including Logistic Regression, KNN, Neural Networks, and CNNs. The app also supports evaluation on the test set with detailed visualizations.

**Target Users** are students, developers, and enthusiasts interested in computer vision, machine learning, and image classification workflows.

**Dataset**: Fashion-MNIST is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Used for benchmarking machine learning algorithms. This is what the data looks like (each class takes three rows):

![](/static/img/fashion-mnist_dataset.png)

![](/static/img/fashion-mnist_dataset.gif)

For more information, see [Zalando's original repo](https://github.com/zalandoresearch/fashion-mnist) which this was pulled from.

🔗 **Try it live**: [fashionmnist-classifier.streamlit.app](https://fashionmnist-classifier.streamlit.app)

<br>

![](/static/img/streamlit_demo_application.png)

---

## 📁 Contents

```bash
├── data/
│   └── data.py          # Dataset loading and preprocessing script
├── models/              # ML model definitions and training logic
├── output/
│   └── visualization.py        # Model performance visualizations
├── saved_models/        # Pre-trained models
├── static/           
│   └── sample_images/   # Sample Fashion MNIST images for testing
├── utils/           
│   └── inference.py     # Utility functions for model inference
├── web/           
│   └── app.py           # Streamlit frontend app
├── LICENSE              # MIT License
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

---

## 🌟 Features

* **Model Selection**: Choose from Logistic Regression, KNN, Neural Network, or CNN models for classification.
* **Image Upload**: Upload your own clothing images in JPG, JPEG, or PNG formats.
* **Live Prediction**: View predicted class and confidence score for uploaded images.
* **Prediction Visualization**: Bar charts display class probabilities for better interpretation.
* **Evaluation Mode**: Evaluate selected model on Fashion MNIST test set with confusion matrix visualization.
* **Preprocessed Samples**: Included sample Fashion MNIST test images and pre-trained models for quick testing.

---

## 🛠️ Installation Instructions

To run the app locally:
1. Clone the repository:
```bash
git clone https://github.com/rmluck/Fashion-MNIST-Classifier.git
cd Fashion-MNIST-Classifier
```
2. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Launch the app:
```bash
streamlit run web/app.py
```

---

## 💡 Usage

**Step 1: Select Model**

Select desired model from sidebar. Choose between Logistic Regression, KNN, Neural Network, or CNN.

**Step 2: Upload Images**

Upload JPG, JPEG, or PNG images of clothing items to get predictions. Find your own or use the provided [sample images](/static/sample_images/).

**Step 3: View Prediction**

View model's prediction and confidence score for selected image(s) along with visualization charting probability for each possible label.

![](/static/img/prediction_results.png)

Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

**Step 4: Evaluate Model**

Run evaluation on the entire Fashion MNIST test set to view confusion matrix and model performance metrics.

![](/static/img/evaluation_metrics.png)

---

## 🚧 Future Improvements

* Add more robust evaluation metrics.
* Add support for batch uploads and batch prediction.
* Provide downloadable prediction reports.
* Add model training interface for user customization.
* Improve UI/UX with dark mode and interactive visualizations.
* Optimize model inference speed with TensorFlow Lite or ONNX.
* Unit test key functions.
* Add image augmentation options to improve model robustness.
* Support mobile-friendly interface.

---

## 🧰 Tech Stack

* Python
* **Frontend**: [Streamlit](https://streamlit.io/)
* **Machine Learning**: TensorFlow/Keras, `scikit-learn`
* **Data Processing**: `numpy`, `PIL` (Pillow)
* **Visualization**: `matplotlib`, `seaborn`, `pandas`
* **Deployment**: Streamlit Community Cloud

## 🙏 Contributions / Acknowledgements

This project was built independently as a portfolio project to demonstrate practical machine learning workflows. Inspired by a course project I did using the Fashion-MNIST dataset.

**Citations**

Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. [arXiv:1708.07747](http://arxiv.org/abs/1708.07747)

## 🪪 License

This project is licensed under the [MIT License](/LICENSE).