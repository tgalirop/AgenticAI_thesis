"""Create diagnostic plots from repeated out-of-fold predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Experiments run non-interactively, including from terminals and CI.  Selecting
# Agg before importing pyplot prevents GUI/backend errors on headless systems.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


def save_diagnostic_plots(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    model_name: str,
    threshold: float,
    figures_root: str | Path,
) -> list[Path]:
    """Save confusion-matrix, PR, and ROC plots for one model.

    Predictions are out-of-fold. With repeated CV, each observation contributes
    once per repeat; the figures therefore describe aggregated validation
    behaviour and never in-sample predictions.
    """

    root = Path(figures_root)
    directories = {
        "confusion_matrix": root / "confusion_matrices",
        "pr_curve": root / "pr_curves",
        "roc_curve": root / "roc_curves",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    true = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    predicted = (scores >= threshold).astype(int)
    outputs: list[Path] = []

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(true, predicted, labels=[0, 1]),
        display_labels=["Normal", "Fraud"],
    ).plot(ax=axis, colorbar=False, cmap="Blues", values_format="d")
    axis.set_title(f"{model_name} — repeated OOF confusion matrix")
    figure.tight_layout()
    path = directories["confusion_matrix"] / f"{model_name}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(true, scores, ax=axis, name=model_name)
    axis.set_title(f"{model_name} — precision-recall curve")
    figure.tight_layout()
    path = directories["pr_curve"] / f"{model_name}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(true, scores, ax=axis, name=model_name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axis.set_title(f"{model_name} — ROC curve")
    figure.tight_layout()
    path = directories["roc_curve"] / f"{model_name}.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)
    return outputs
