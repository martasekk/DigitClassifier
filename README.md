Number Classifier App
=====================

A small GUI app for recognizing handwritten digits (0–9).
Built with CustomTkinter for the interface, TensorFlow/Keras for the model,
and Pillow for image drawing.

Features:
- Draw a digit on the canvas
- Model predicts the digit when you release the mouse
- Clear button to reset the canvas
- Trains an MNIST model if none is found

Installation:
1. Clone this repo
2. (Optional) create a virtual environment
3. Install dependencies:
   pip install -r requirements.txt

Usage:
   python main.py

Files:
- main.py: main application
- mnist_model.keras: saved model (created after training)
- requirements.txt: dependencies
- .gitignore: ignore rules
- README.txt: this file
