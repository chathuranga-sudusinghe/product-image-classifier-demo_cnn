from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# Add project root to sys.path so pytest can import app.main
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.main import CLASS_NAMES, predict_image, preprocess_image  # noqa: E402


class DummyModel:
    """
    Fake model for testing predict_image without loading the real model.
    """

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        probabilities = np.zeros((1, len(CLASS_NAMES)), dtype=np.float32)
        probabilities[0, 0] = 0.05
        probabilities[0, 1] = 0.10
        probabilities[0, 2] = 0.15
        probabilities[0, 3] = 0.70
        return probabilities


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """
    Create a random RGB image for testing.
    """
    image_array = np.random.randint(0, 256, size=(40, 40, 3), dtype=np.uint8)
    return Image.fromarray(image_array)


def test_preprocess_image_returns_correct_shape(sample_pil_image: Image.Image) -> None:
    """
    Test that preprocessing returns the correct batch shape and normalized range.
    """
    processed = preprocess_image(sample_pil_image)

    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 32, 32, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_predict_image_returns_probability_dictionary(
    monkeypatch: pytest.MonkeyPatch,
    sample_pil_image: Image.Image,
) -> None:
    """
    Test that predict_image returns a dictionary of class probabilities.
    """
    import app.main as main_module

    monkeypatch.setattr(main_module, "model", DummyModel())

    result = predict_image(sample_pil_image)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(CLASS_NAMES)
    assert all(isinstance(value, float) for value in result.values())
    assert result[CLASS_NAMES[3]] == pytest.approx(0.70, rel=1e-3)


def test_predict_image_raises_error_for_none_input() -> None:
    """
    Test that None input raises ValueError.
    """
    with pytest.raises(ValueError, match="Please upload an image before prediction."):
        predict_image(None)