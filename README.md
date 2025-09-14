# Alphabet-Classifier

🅰️ Alphabet Recognition using Neural Networks
📌 Project Overview

This project implements a Neural Network Architecture using ReLU and Softmax activation functions to recognize handwritten alphabets.
The model is trained on a CSV dataset of handwritten letters.
It preprocesses input images and predicts the alphabet with high accuracy.
The system can take images or drawings as input and classify them into A–Z.

👉 A simple but powerful project demonstrating deep learning’s potential in handwriting recognition.

###########################################################################################################

⚙️ How It Works

The dataset (A_Z Handwritten Data.csv) is loaded using Pandas.
Input images are normalized and reshaped for training.
A Sequential Neural Network is built using TensorFlow/Keras:
Hidden layers → ReLU activation
Output layer → Softmax activation for 26 classes (A–Z)

The model is trained, validated, and saved as my_model_kushagraDwivedi.keras

Give it an input image, the trained model predicts the correct alphabet.

###########################################################################################################

🧠 Neural Network Architecture

Input Layer: Flattened 28×28 pixel image

Hidden Layers: Dense layers with ReLU activation

Output Layer: Dense layer with Softmax activation → 26 outputs (A–Z)

###########################################################################################################

💡 Real-World Application

This project can be extended to:
✔️ Recognize scribbled answers in school/college answer sheets
✔️ Assist teachers in automatic evaluation of student handwriting
✔️ Serve as the foundation for OCR (Optical Character Recognition) systems

###########################################################################################################

📂 Tech Stack

Python

NumPy & Pandas – Data handling
TensorFlow/Keras – Neural Network
Matplotlib – Visualization
OpenCV (cv2) – Image preprocessing

###########################################################################################################

🚀 How to Run

Clone this repository:

git clone https://github.com/your-username/alphabet-recognition.git
cd alphabet-recognition

First Install the dataset from Kaggle using the below link:
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

Install dependencies:
pip install numpy pandas tensorflow matplotlib


Test with an image:
provide the image path to the model for evaluating its correctness










