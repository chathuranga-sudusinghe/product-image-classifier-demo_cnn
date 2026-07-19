from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.train import CLASS_NAMES, save_evaluation_evidence


REQUIRED_METRIC_KEYS = {
    "dataset_name",
    "random_seed",
    "train_subset_size",
    "validation_subset_size",
    "test_subset_size",
    "epochs",
    "batch_size",
    "tensorflow_version",
    "final_test_loss",
    "final_test_accuracy",
    "model_parameter_count",
}


def save_synthetic_evidence(tmp_path: Path) -> dict[str, Path]:
    y_true = np.repeat(np.arange(10), 2)
    y_pred = y_true.copy()
    y_pred[1] = 1

    return save_evaluation_evidence(
        history={
            "loss": [np.float32(1.2), np.float32(0.8)],
            "accuracy": [np.float32(0.5), np.float32(0.7)],
        },
        y_true=y_true,
        y_pred=y_pred,
        test_loss=np.float32(0.75),
        test_accuracy=np.float32(0.8),
        model_parameter_count=np.int64(1234),
        tensorflow_version="test-version",
        train_size=100,
        validation_size=20,
        test_size=len(y_true),
        evaluation_dir=tmp_path,
    )


def test_evaluation_files_are_created_with_required_metrics(tmp_path: Path) -> None:
    paths = save_synthetic_evidence(tmp_path)

    assert all(path.is_file() for path in paths.values())
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert REQUIRED_METRIC_KEYS <= metrics.keys()


def test_confusion_matrix_has_ten_by_ten_shape(tmp_path: Path) -> None:
    paths = save_synthetic_evidence(tmp_path)

    confusion_matrix = np.loadtxt(paths["confusion_matrix"], delimiter=",")
    assert confusion_matrix.shape == (10, 10)


def test_classification_report_contains_every_cifar10_class(tmp_path: Path) -> None:
    paths = save_synthetic_evidence(tmp_path)

    report = json.loads(
        paths["classification_report"].read_text(encoding="utf-8")
    )
    assert set(report) == set(CLASS_NAMES)
    for class_metrics in report.values():
        assert set(class_metrics) == {
            "precision",
            "recall",
            "f1-score",
            "support",
        }


def test_json_outputs_are_serializable(tmp_path: Path) -> None:
    paths = save_synthetic_evidence(tmp_path)

    for name in ("metrics", "training_history", "classification_report"):
        loaded = json.loads(paths[name].read_text(encoding="utf-8"))
        json.dumps(loaded)
