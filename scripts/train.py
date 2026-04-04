from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

from core.logger import setup_logger

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
RANDOM_STATE = 42
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
EPOCHS = 5
BATCH_SIZE = 64
TRAIN_SUBSET_SIZE = 10000
TEST_SUBSET_SIZE = 2000

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logger = setup_logger(Path(__file__).stem) # Create a logger with the name of the current script
                                           # .stem gives the filename without the extension, 

def load_dataset() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Load the raw CIFAR-10 dataset only.
    """
    logger.info("Loading CIFAR-10 dataset...")

    dataset = cifar10.load_data()

    logger.info("CIFAR-10 dataset loaded successfully.")
    return dataset


def prepare_subset(
    dataset: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract train/test arrays and keep a smaller subset for faster training.
    """
    (x_train, y_train), (x_test, y_test) = dataset

    # Flatten labels from shape (n, 1) to (n,) 1 = one type pf sample, n = number of samples
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    logger.info("Original training samples: %s", len(x_train))
    logger.info("Original test samples: %s", len(x_test))

    x_train = x_train[:TRAIN_SUBSET_SIZE] # Keep only the first TRAIN_SUBSET_SIZE samples for training
    y_train = y_train[:TRAIN_SUBSET_SIZE]
    x_test = x_test[:TEST_SUBSET_SIZE]
    y_test = y_test[:TEST_SUBSET_SIZE]

    logger.info("Using training subset: %s", len(x_train)) # %s is a placeholder 
    logger.info("Using test subset: %s", len(x_test))

    return x_train, y_train, x_test, y_test


def preprocess_images(
    x_train: np.ndarray,  # ndarray = n-dimensional array
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize image pixel values to the range [0, 1].
    """
    logger.info("Preprocessing images...")

    x_train = x_train.astype("float32") / 255.0 # 255 and 255.0 are the same, but 255.0 is a float, only for clarity
    x_test = x_test.astype("float32") / 255.0

    logger.info("Image preprocessing completed.")
    return x_train, x_test


def build_model() -> keras.Model:
    """
    Build a small CNN for CIFAR-10 classification.
    """
    logger.info("Building CNN model...")

    model = keras.Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"), # same -> input size = output size
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("CNN model built successfully.")
    return model


def save_artifacts(model: keras.Model) -> None:
    """
    Save the trained model and class-name metadata.
    """
    logger.info("Saving model artifacts...")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "model.keras"
    labels_path = ARTIFACTS_DIR / "class_names.json"

    model.save(model_path)

    with labels_path.open("w", encoding="utf-8") as file:
        json.dump(CLASS_NAMES, file, indent=2)

    logger.info("Artifacts saved successfully in %s", ARTIFACTS_DIR)


def main() -> None:
    """
    Run the full image classification training pipeline.
    """
    logger.info("Starting training pipeline...")

    np.random.seed(RANDOM_STATE)
    keras.utils.set_random_seed(RANDOM_STATE)

    dataset = load_dataset()

    x_train, y_train, x_test, y_test = prepare_subset(dataset)

    x_train, x_test = preprocess_images(
        x_train=x_train,
        x_test=x_test,
    )

    model = build_model()

    logger.info("Starting model training...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    logger.info("Evaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    logger.info("Test loss: %.4f", loss)
    logger.info("Test accuracy: %.4f", accuracy)

    save_artifacts(model)
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()