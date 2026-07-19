from __future__ import annotations

import numpy as np

from scripts.train import prepare_subsets, select_subset_indices


def make_labels(samples_per_class: int, num_classes: int = 4) -> np.ndarray:
    """Create balanced labels in the same column shape returned by CIFAR-10."""
    return np.repeat(np.arange(num_classes), samples_per_class).reshape(-1, 1)


def test_subset_selection_is_deterministic_for_same_seed() -> None:
    y_train = make_labels(samples_per_class=20)
    y_test = make_labels(samples_per_class=10)
    first = select_subset_indices(y_train, y_test, 24, 12, 16, seed=42)
    second = select_subset_indices(y_train, y_test, 24, 12, 16, seed=42)
    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    np.testing.assert_array_equal(first.test, second.test)


def test_different_seeds_produce_different_selections() -> None:
    y_train = make_labels(samples_per_class=20)
    y_test = make_labels(samples_per_class=10)
    first = select_subset_indices(y_train, y_test, 24, 12, 16, seed=1)
    second = select_subset_indices(y_train, y_test, 24, 12, 16, seed=2)
    assert not np.array_equal(first.train, second.train)
    assert not np.array_equal(first.validation, second.validation)
    assert not np.array_equal(first.test, second.test)


def test_train_and_validation_indices_are_disjoint_and_have_expected_sizes() -> None:
    indices = select_subset_indices(
        make_labels(samples_per_class=20),
        make_labels(samples_per_class=10),
        train_size=24,
        validation_size=12,
        test_size=16,
        seed=42,
    )
    assert len(indices.train) == 24
    assert len(indices.validation) == 12
    assert len(indices.test) == 16
    assert np.intersect1d(indices.train, indices.validation).size == 0


def test_each_subset_is_reasonably_class_balanced() -> None:
    y_train = make_labels(samples_per_class=20).reshape(-1)
    y_test = make_labels(samples_per_class=10).reshape(-1)
    indices = select_subset_indices(
        y_train,
        y_test,
        train_size=25,
        validation_size=13,
        test_size=17,
        seed=42,
    )
    for labels, selected in (
        (y_train, indices.train),
        (y_train, indices.validation),
        (y_test, indices.test),
    ):
        counts = np.bincount(labels[selected], minlength=4)
        assert counts.max() - counts.min() <= 1


def test_test_subset_comes_only_from_test_partition() -> None:
    y_train = make_labels(samples_per_class=20)
    y_test = make_labels(samples_per_class=10)
    x_train = np.arange(len(y_train))
    x_test = np.arange(1000, 1000 + len(y_test))
    selected_train, _, selected_validation, _, selected_test, _ = prepare_subsets(
        ((x_train, y_train), (x_test, y_test)),
        train_size=24,
        validation_size=12,
        test_size=16,
        seed=42,
    )
    assert np.all(selected_train < 1000)
    assert np.all(selected_validation < 1000)
    assert np.all(selected_test >= 1000)
