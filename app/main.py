from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from tensorflow import keras

from core.logger import setup_logger

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "model.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
INPUT_SIZE = (32, 32)

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logger = setup_logger(Path(__file__).stem)

# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

logger.info("Loading trained model...")
model = keras.models.load_model(MODEL_PATH)

logger.info("Loading class names...")
with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
    CLASS_NAMES: list[str] = json.load(file)

    """ # Alternative way to load class names without using 'with' statement
        file = CLASS_NAMES_PATH.open("r", encoding="utf-8")
        CLASS_NAMES = json.load(file)
        file.close()
    """

logger.info("Artifacts loaded successfully.")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize and normalize a single uploaded image for prediction.
    """
    logger.info("Preprocessing uploaded image...")

    image = image.convert("RGB")
    image = image.resize(INPUT_SIZE)

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    logger.info("Image preprocessing completed.")
    return image_array


def predict_image(image: Image.Image) -> dict[str, float]:
    """
    Predict class probabilities for an uploaded image.
    """
    if image is None:
        raise ValueError("Please upload an image before prediction.")

    processed_image = preprocess_image(image)

    logger.info("Running model prediction...")

    # model.predict returns a batch of predictions
    # verbose=0 prevents unnecessary messages in the console during prediction
    # [0] gets the first (and only) prediction from the batch
    probabilities = model.predict(processed_image, verbose=0)[0] 
                                                                 

    prediction_result = {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities) # zip pairs each class name with its corresponding probability, creating a dictionary
    }      

    """  # without dictionary comprehension

       prediction_result = {}

      for i in range(len(CLASS_NAMES)):
         class_name = CLASS_NAMES[i]
         probability = probabilities[i]
         prediction_result[class_name] = float(probability)

    """

    logger.info("Prediction completed successfully.")
    return prediction_result

# ---------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Product Image Classifier Demo",
    description="Upload an image to predict its class using the trained CNN model.",
)

# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=7860)