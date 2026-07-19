from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
VALIDATION_SUBSET_SIZE = 2000
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


@dataclass(frozen=True)
class SubsetIndices:
    """Indices selected from the original CIFAR-10 train and test partitions."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logger = setup_logger(Path(__file__).stem) # Create a logger with the name of the current script
                                           # .stem gives the filename without the extension, 

def load_dataset() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Load the raw CIFAR-10 dataset only.
    """
    from tensorflow.keras.datasets import cifar10

    logger.info("Loading CIFAR-10 dataset...")

    dataset = cifar10.load_data()

    logger.info("CIFAR-10 dataset loaded successfully.")
    return dataset


def _balanced_counts(subset_size: int, num_classes: int) -> np.ndarray:
    """Distribute a subset size across classes with at most one sample difference."""
    if subset_size < 0:
        raise ValueError("Subset sizes must be non-negative.")
    if num_classes < 1:
        raise ValueError("At least one class is required.")

    counts = np.full(num_classes, subset_size // num_classes, dtype=np.int64)
    counts[: subset_size % num_classes] += 1
    return counts


def _select_balanced_partition_indices(
    labels: np.ndarray,
    subset_sizes: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    """Select disjoint, approximately class-balanced subsets from one partition."""
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)
    if classes.size == 0:
        raise ValueError("Cannot select subsets from an empty label array.")

    counts_by_subset = [
        _balanced_counts(subset_size, len(classes)) for subset_size in subset_sizes
    ]
    selected_by_subset: list[list[np.ndarray]] = [[] for _ in subset_sizes]

    for class_position, class_label in enumerate(classes):
        class_indices = np.flatnonzero(labels == class_label)
        required = sum(int(counts[class_position]) for counts in counts_by_subset)
        if required > len(class_indices):
            raise ValueError(
                f"Class {class_label!r} has {len(class_indices)} samples, "
                f"but {required} are required."
            )

        shuffled_indices = rng.permutation(class_indices)
        offset = 0
        for subset_position, counts in enumerate(counts_by_subset):
            count = int(counts[class_position])
            selected_by_subset[subset_position].append(
                shuffled_indices[offset : offset + count]
            )
            offset += count

    selected_indices: list[np.ndarray] = []
    for class_selections in selected_by_subset:
        indices = np.concatenate(class_selections).astype(np.int64, copy=False)
        rng.shuffle(indices)
        selected_indices.append(indices)
    return tuple(selected_indices)


def select_subset_indices(
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_size: int = TRAIN_SUBSET_SIZE,
    validation_size: int = VALIDATION_SUBSET_SIZE,
    test_size: int = TEST_SUBSET_SIZE,
    seed: int = RANDOM_STATE,
) -> SubsetIndices:
    """Select reproducible, balanced indices without mixing dataset partitions."""
    rng = np.random.default_rng(seed)
    train_indices, validation_indices = _select_balanced_partition_indices(
        labels=y_train,
        subset_sizes=(train_size, validation_size),
        rng=rng,
    )
    (test_indices,) = _select_balanced_partition_indices(
        labels=y_test,
        subset_sizes=(test_size,),
        rng=rng,
    )
    return SubsetIndices(
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
    )


def prepare_subsets(
    dataset: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    train_size: int = TRAIN_SUBSET_SIZE,
    validation_size: int = VALIDATION_SUBSET_SIZE,
    test_size: int = TEST_SUBSET_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build train/validation subsets from training data and a separate test subset."""
    (x_train_full, y_train_full), (x_test_full, y_test_full) = dataset
    y_train_full = np.asarray(y_train_full).reshape(-1)
    y_test_full = np.asarray(y_test_full).reshape(-1)

    if len(x_train_full) != len(y_train_full):
        raise ValueError("Training images and labels must have the same length.")
    if len(x_test_full) != len(y_test_full):
        raise ValueError("Test images and labels must have the same length.")

    logger.info("Original training samples: %s", len(x_train_full))
    logger.info("Original test samples: %s", len(x_test_full))
    indices = select_subset_indices(
        y_train=y_train_full,
        y_test=y_test_full,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        seed=seed,
    )

    x_train = x_train_full[indices.train]
    y_train = y_train_full[indices.train]
    x_validation = x_train_full[indices.validation]
    y_validation = y_train_full[indices.validation]
    x_test = x_test_full[indices.test]
    y_test = y_test_full[indices.test]

    logger.info("Using training subset: %s", len(x_train))
    logger.info("Using validation subset: %s", len(x_validation))
    logger.info("Using test subset: %s", len(x_test))
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def preprocess_images(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize image pixel values to the range [0, 1]."""
    logger.info("Preprocessing images...")
    x_train = x_train.astype("float32") / 255.0
    x_validation = x_validation.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    logger.info("Image preprocessing completed.")
    return x_train, x_validation, x_test


def build_model() -> Any:
    """
    Build a small CNN for CIFAR-10 classification.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

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


def save_artifacts(model: Any) -> None:
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
    from tensorflow import keras

    logger.info("Starting training pipeline...")

    np.random.seed(RANDOM_STATE)
    keras.utils.set_random_seed(RANDOM_STATE)

    dataset = load_dataset()

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = prepare_subsets(dataset)

    x_train, x_validation, x_test = preprocess_images(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
    )

    model = build_model()

    logger.info("Starting model training...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    logger.info("Evaluating model once on the untouched test subset...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    logger.info("Test loss: %.4f", loss)
    logger.info("Test accuracy: %.4f", accuracy)

    save_artifacts(model)
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
