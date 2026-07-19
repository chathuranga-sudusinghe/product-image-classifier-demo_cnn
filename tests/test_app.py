from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import app.main as main_module
from app.main import (
    CLASS_NAMES_PATH,
    MODEL_PATH,
    load_artifacts,
    predict_image,
    preprocess_image,
    validate_class_names,
)


TEST_CLASS_NAMES = [
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


class DummyModel:
    """Small injectable prediction model used by unit tests."""

    def __init__(self, probabilities: np.ndarray | None = None) -> None:
        if probabilities is None:
            probabilities = np.array(
                [0.05, 0.10, 0.15, 0.70, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
        self.probabilities = probabilities

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert batch.shape == (1, 32, 32, 3)
        assert verbose == 0
        return np.expand_dims(self.probabilities, axis=0)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    image_array = np.full((40, 40, 3), 128, dtype=np.uint8)
    return Image.fromarray(image_array)


def test_import_does_not_load_model_or_create_interface() -> None:
    assert "model" not in vars(main_module)
    assert "demo" not in vars(main_module)


def test_preprocess_image_returns_correct_shape(sample_pil_image: Image.Image) -> None:
    processed = preprocess_image(sample_pil_image)

    assert processed.shape == (1, 32, 32, 3)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0


def test_predict_image_uses_injected_model_and_returns_python_floats(
    sample_pil_image: Image.Image,
) -> None:
    result = predict_image(sample_pil_image, DummyModel(), TEST_CLASS_NAMES)

    assert set(result) == set(TEST_CLASS_NAMES)
    assert all(isinstance(value, float) for value in result.values())
    assert result[TEST_CLASS_NAMES[3]] == pytest.approx(0.70, rel=1e-3)


def test_predict_image_raises_error_for_none_input() -> None:
    with pytest.raises(ValueError, match="Please upload an image"):
        predict_image(None, DummyModel(), TEST_CLASS_NAMES)


def test_predict_image_rejects_non_image_input() -> None:
    with pytest.raises(TypeError, match="must be a PIL image"):
        predict_image("not-an-image", DummyModel(), TEST_CLASS_NAMES)  # type: ignore[arg-type]


def test_predict_image_rejects_output_label_mismatch(
    sample_pil_image: Image.Image,
) -> None:
    model = DummyModel(np.zeros(9, dtype=np.float32))

    with pytest.raises(ValueError, match="output length does not match"):
        predict_image(sample_pil_image, model, TEST_CLASS_NAMES)


def test_validate_class_names_rejects_inconsistent_metadata() -> None:
    duplicate_names = TEST_CLASS_NAMES[:-1] + [TEST_CLASS_NAMES[0]]

    with pytest.raises(ValueError, match="must be unique"):
        validate_class_names(duplicate_names)


def test_load_artifacts_checks_required_files(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.keras"
    missing_labels = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_artifacts(missing_model, missing_labels)

    existing_model = tmp_path / "model.keras"
    existing_model.touch()
    with pytest.raises(FileNotFoundError, match="Class names file not found"):
        load_artifacts(existing_model, missing_labels)


def test_load_artifacts_validates_labels_before_loading_model(tmp_path: Path) -> None:
    model_path = tmp_path / "model.keras"
    labels_path = tmp_path / "class_names.json"
    model_path.touch()
    labels_path.write_text(
        json.dumps(TEST_CLASS_NAMES[:-1] + [TEST_CLASS_NAMES[0]]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be unique"):
        load_artifacts(model_path, labels_path)


def test_load_artifacts_returns_injected_tensorflow_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.keras"
    labels_path = tmp_path / "class_names.json"
    model_path.touch()
    labels_path.write_text(json.dumps(TEST_CLASS_NAMES), encoding="utf-8")
    expected_model = object()

    fake_keras = types.SimpleNamespace(
        models=types.SimpleNamespace(load_model=lambda path: expected_model)
    )
    fake_tensorflow = types.ModuleType("tensorflow")
    fake_tensorflow.keras = fake_keras
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)

    model, class_names = load_artifacts(model_path, labels_path)

    assert model is expected_model
    assert class_names == TEST_CLASS_NAMES


def test_load_artifacts_smoke() -> None:
    pytest.importorskip("tensorflow")

    model, class_names = load_artifacts(MODEL_PATH, CLASS_NAMES_PATH)

    assert class_names == TEST_CLASS_NAMES
    assert model.output_shape[-1] == len(class_names)
