# Product Image Classifier Demo

A simple CNN-based image classification demo built with TensorFlow/Keras and Gradio.  
This project trains a small convolutional neural network on a CIFAR-10 subset and provides a browser-based interface for image prediction.

## Project Goal

The goal of this mini project is to practice the end-to-end workflow of a computer vision application:

- load and prepare image data
- train a CNN model
- save model artifacts
- build an inference app
- run predictions through a Gradio web interface

## Tools

- Python 3.12
- TensorFlow / Keras
- NumPy
- Pillow
- Gradio
- Pytest
- Docker
- GitHub Actions

## Project Structure

```text
product-image-classifier-demo/
├── app/
│   └── main.py
├── artifacts/
│   ├── model.keras
│   └── class_names.json
├── core/
│   └── logger.py
├── logs/
│   └── app.log
├── scripts/
│   └── train.py
├── tests/
│   └── test_app.py
├── infra/
│   └── terraform/
│       └── main.tf
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

### Training flow

- Load the CIFAR-10 dataset
- Keep a smaller subset for faster mini-project training
- Normalize image pixel values
- Train a small CNN model
- Evaluate the model on test data
- Save the trained model and class names

### Inference flow

- User uploads an image in the Gradio app
- The image is converted to RGB
- The image is resized to `32 x 32`
- Pixel values are normalized to the range `[0, 1]`
- The trained model predicts class probabilities
- The app shows the top predicted classes

## CIFAR-10 Classes

The model predicts one of the following classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Setup Instructions

### 1. Create and activate virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
### 2. Install dependencies
```
pip install -r requirements.txt
```

### Run the Gradio App
```
python app/main.py
```
## Output

The app shows:

- top predicted class
- confidence scores for the best classes

## Current Limitation

This is a mini project built for skill polishing, so the model accuracy is limited because:

- a small CNN is used
- only a subset of CIFAR-10 is used
- training is intentionally lightweight for fast experimentation

## Future Improvements

- train on the full CIFAR-10 dataset
- improve model architecture
- add data augmentation
- add automated tests
- containerize with Docker
- add CI workflow
- deploy as a small cloud demo

## Author

Chathuranga Sudusinghe
