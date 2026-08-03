"""Compute fraud-classification metrics from scores and a fixed threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate threshold-dependent and ranking metrics consistently.

    PR-AUC is represented by Average Precision, sklearn's step-wise summary of
    the precision-recall curve.  It is the primary metric because Accuracy can be
    misleading when fraud represents roughly 0.1% of the original PaySim data.
    """

    true = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if true.shape != scores.shape:
        raise ValueError("y_true and y_score must have identical shapes")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if np.unique(true).size != 2:
        raise ValueError("Both classes must be present to compute ROC-AUC and PR-AUC")

    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy": accuracy_score(true, predicted),
        "recall": recall_score(true, predicted, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(true, predicted, zero_division=0),
        "f1": f1_score(true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(true, scores),
        "pr_auc": average_precision_score(true, scores),
        "balanced_accuracy": balanced_accuracy_score(true, predicted),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
