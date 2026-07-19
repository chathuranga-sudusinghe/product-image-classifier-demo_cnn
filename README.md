# CIFAR-10 Image Classifier

A compact computer vision project that trains a small convolutional neural network (CNN) to classify images into the ten CIFAR-10 classes. It includes a reproducible training pipeline, saved evaluation evidence, a Gradio inference interface, lightweight tests, Docker support, and a focused GitHub Actions workflow.

This is a small supporting CNN portfolio project for demonstrating core image-classification practices. It is not intended to be a production or enterprise system.

## Purpose and scope

The project demonstrates a complete but intentionally limited workflow:

- reproducible, class-balanced subset selection with separate training, validation, and test data;
- a straightforward CNN trained on normalized 32 x 32 RGB images;
- final evaluation on a held-out subset of the official CIFAR-10 test partition;
- versioned model, label, and evaluation artifacts;
- local inference through a simple Gradio interface;
- automated compile and test checks.

## Key features

- Seeded NumPy-based subset selection with no additional splitting dependency
- Disjoint training and validation subsets drawn only from the CIFAR-10 training partition
- Test data used only once for final evaluation after training
- Saved metrics, training history, confusion matrix, and per-class classification report
- Validated model and class-label loading
- Injectable prediction logic that is testable without loading the real model
- CPU-compatible Docker image and minimal GitHub Actions CI

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) contains 60,000 color images at 32 x 32 pixels across ten classes. Its standard split has 50,000 training images and 10,000 test images.

This project uses seeded, class-balanced subsets:

- 10,000 training images from the official training partition
- 2,000 validation images from the official training partition
- 2,000 test images from the official test partition

Training and validation indices are disjoint. The test subset is never passed to `model.fit()`.

On the first training run, Keras downloads CIFAR-10 automatically to its default user cache, normally `~/.keras/datasets`. The repository does not require or maintain a project-level `data/` directory.

## Verified results

The tracked evaluation evidence records the following final test results:

| Metric | Result |
| --- | ---: |
| Test accuracy | 57.55% |
| Test loss | 1.1888 |
| Model parameters | 1,143,242 |

These results were produced with random seed 42, 5 epochs, a batch size of 64, and TensorFlow 2.21.0. Small numerical differences can occur across hardware and TensorFlow execution environments.

## Supported classes

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## Project structure

```text
.
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- app/
|   |-- __init__.py
|   `-- main.py
|-- artifacts/
|   |-- class_names.json
|   `-- model.keras
|-- core/
|   |-- __init__.py
|   `-- logger.py
|-- evaluation/
|   |-- classification_report.json
|   |-- confusion_matrix.csv
|   |-- metrics.json
|   `-- training_history.json
|-- scripts/
|   |-- __init__.py
|   `-- train.py
|-- tests/
|   |-- __init__.py
|   |-- test_app.py
|   |-- test_evaluation.py
|   `-- test_train.py
|-- .dockerignore
|-- .gitignore
|-- Dockerfile
|-- README.md
`-- requirements.txt
```

## Setup on Linux

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.12 is used by the Docker image and CI workflow.

## Train the model

```bash
python -m scripts.train
```

This command downloads CIFAR-10 through Keras when needed, trains the CNN, and overwrites the tracked files in `artifacts/` and `evaluation/`. Training is not required to run the repository's pre-trained inference demo.

## Run the application

```bash
python -m app.main
```

Open `http://localhost:7860`, upload an image, and review the top three predictions. The application converts the image to RGB, resizes it to 32 x 32, and normalizes pixel values consistently with training.

## Run the tests

```bash
python -m pytest -q
```

The tests cover dataset subset selection, evaluation output generation, preprocessing, prediction metadata validation, interface construction, and artifact-loading behavior.

## Docker

Build the CPU-compatible image from the repository root:

```bash
docker build -t cifar10-image-classifier .
```

Run the container:

```bash
docker run --rm -p 7860:7860 cifar10-image-classifier
```

Then open `http://localhost:7860`.

## Evaluation evidence

The tracked `evaluation/` directory contains:

- `metrics.json`: dataset, seed, subset sizes, training configuration, TensorFlow version, final test metrics, and parameter count
- `training_history.json`: per-epoch training and validation loss and accuracy
- `classification_report.json`: precision, recall, F1-score, and support for every CIFAR-10 class
- `confusion_matrix.csv`: 10 x 10 final-test confusion matrix

The tracked `artifacts/model.keras` and `artifacts/class_names.json` allow a fresh clone with installed dependencies to run inference without retraining.

## Limitations

- Training uses balanced subsets rather than the complete CIFAR-10 dataset and runs for only five epochs.
- The 57.55% test accuracy is appropriate as learning evidence, but it is not competitive with modern image-classification systems.
- Uploaded photographs may differ substantially from CIFAR-10's low-resolution data; resizing every input to 32 x 32 can remove useful detail.
- The model returns class probabilities only. It does not detect out-of-distribution images or provide calibrated confidence guarantees.
- Reproducibility is seeded, but exact floating-point results may vary slightly by platform and TensorFlow backend.
