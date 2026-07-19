from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.logger import setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"

INPUT_SIZE = (32, 32)
EXPECTED_CLASS_COUNT = 10

logger = setup_logger(Path(__file__).stem)


def validate_class_names(class_names: Any) -> list[str]:
    """Validate the class-label metadata expected by the CIFAR-10 model."""
    if not isinstance(class_names, list):
        raise ValueError("Class names metadata must be a JSON list.")
    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Class names metadata must contain {EXPECTED_CLASS_COUNT} labels."
        )
    if any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError("Class names must be non-empty strings.")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Class names must be unique.")
    return list(class_names)


def load_artifacts(
    model_path: Path = MODEL_PATH,
    class_names_path: Path = CLASS_NAMES_PATH,
) -> tuple[Any, list[str]]:
    """Load and validate the trained model and its class-label metadata."""
    model_path = Path(model_path)
    class_names_path = Path(class_names_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not class_names_path.is_file():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")

    try:
        with class_names_path.open("r", encoding="utf-8") as file:
            class_names = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Class names file is not valid JSON: {class_names_path}"
        ) from error

    validated_class_names = validate_class_names(class_names)

    from tensorflow import keras

    logger.info("Loading trained model from %s", model_path)
    model = keras.models.load_model(model_path)
    logger.info("Model and class names loaded successfully.")
    return model, validated_class_names


def preprocess_image(image: Image.Image | None) -> np.ndarray:
    """Validate, resize, and normalize one uploaded image for prediction."""
    if image is None:
        raise ValueError("Please upload an image before prediction.")
    if not isinstance(image, Image.Image):
        raise TypeError("Uploaded image must be a PIL image.")

    converted_image = image.convert("RGB")
    resized_image = converted_image.resize(INPUT_SIZE)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(
    image: Image.Image | None,
    model: Any,
    class_names: list[str],
) -> dict[str, float]:
    """Predict class probabilities using injected model and label metadata."""
    processed_image = preprocess_image(image)
    validated_class_names = validate_class_names(class_names)

    logger.info("Running model prediction...")
    prediction_batch = np.asarray(model.predict(processed_image, verbose=0))
    if prediction_batch.ndim != 2 or prediction_batch.shape[0] != 1:
        raise ValueError(
            "Model prediction must contain exactly one two-dimensional batch."
        )

    probabilities = prediction_batch[0]
    if len(probabilities) != len(validated_class_names):
        raise ValueError(
            "Model prediction output length does not match the class-name count."
        )

    result = {
        class_name: float(probability)
        for class_name, probability in zip(validated_class_names, probabilities)
    }
    logger.info("Prediction completed successfully.")
    return result


def create_interface(model: Any, class_names: list[str]) -> Any:
    """Create the Gradio interface for already-loaded artifacts."""
    validated_class_names = validate_class_names(class_names)

    import gradio as gr

    prediction_function = partial(
        predict_image,
        model=model,
        class_names=validated_class_names,
    )
    return gr.Interface(
        fn=prediction_function,
        inputs=gr.Image(type="pil", label="Upload an image"),
        outputs=gr.Label(num_top_classes=3, label="Prediction"),
        title="Product Image Classifier Demo",
        description="Upload an image to predict its class using the trained CNN model.",
    )


def main() -> None:
    """Load artifacts and launch the Gradio application."""
    model, class_names = load_artifacts()
    demo = create_interface(model, class_names)
    logger.info("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
