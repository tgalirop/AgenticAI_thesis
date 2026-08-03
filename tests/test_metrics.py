"""Tests for classification metric calculations."""

import numpy as np
import pytest

from agenticai_thesis.modeling.metrics import compute_classification_metrics


def test_metrics_from_known_predictions() -> None:
    """The confusion-matrix terms and derived rates must remain consistent."""

    metrics = compute_classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.8, 0.9, 0.2]),
        threshold=0.5,
    )
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["accuracy"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5


def test_metrics_require_both_classes() -> None:
    with pytest.raises(ValueError, match="Both classes"):
        compute_classification_metrics(np.array([0, 0]), np.array([0.1, 0.2]))


def test_metrics_reject_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        compute_classification_metrics(np.array([0, 1]), np.array([0.1, 0.9]), threshold=1.1)
