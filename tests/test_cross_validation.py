"""Tests for immutable shared cross-validation fold definitions."""

import numpy as np
import pytest

from agenticai_thesis.modeling.cross_validation import CrossValidationFoldProvider


def test_fold_provider_is_deterministic_and_complete() -> None:
    y = np.asarray([0, 1] * 15)
    provider = CrossValidationFoldProvider(folds=3, repeats=2, random_seed=42)
    first = provider.create(y)
    second = provider.create(y)

    assert len(first.splits) == 6
    assert first.target_sha256 == second.target_sha256
    for left, right in zip(first.splits, second.splits, strict=True):
        np.testing.assert_array_equal(left.train_indices, right.train_indices)
        np.testing.assert_array_equal(left.validation_indices, right.validation_indices)
        assert set(left.train_indices).isdisjoint(left.validation_indices)
        assert len(left.train_indices) + len(left.validation_indices) == len(y)


def test_fold_indices_are_read_only() -> None:
    fold_set = CrossValidationFoldProvider(folds=2, repeats=1, random_seed=42).create(
        np.asarray([0, 1] * 5)
    )
    with pytest.raises(ValueError, match="read-only"):
        fold_set.splits[0].train_indices[0] = 999


def test_fold_set_rejects_different_target_order() -> None:
    y = np.asarray([0, 0, 0, 1, 1, 1])
    fold_set = CrossValidationFoldProvider(folds=3, repeats=1, random_seed=42).create(y)
    reordered = y[::-1]
    with pytest.raises(ValueError, match="fingerprint"):
        fold_set.validate_target(reordered)


def test_fold_provider_requires_enough_minority_observations() -> None:
    with pytest.raises(ValueError, match="at least one observation per fold"):
        CrossValidationFoldProvider(folds=3, repeats=1, random_seed=42).create(
            np.asarray([0, 0, 0, 0, 1, 1])
        )

