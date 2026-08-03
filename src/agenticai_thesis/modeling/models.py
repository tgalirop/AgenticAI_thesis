"""Create deterministic conventional fraud-detection estimators."""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


SUPPORTED_MODELS = ("logistic_regression", "decision_tree", "random_forest")


def create_estimator(
    name: str,
    *,
    random_seed: int,
    parameters: Mapping[str, Any] | None = None,
) -> ClassifierMixin:
    """Create one estimator with imbalance handling and reproducible randomness."""

    params = dict(parameters or {})
    if name == "logistic_regression":
        # Class weights are calculated from each training fold, so validation
        # labels never influence the fitted decision boundary.
        return LogisticRegression(
            class_weight="balanced",
            random_state=random_seed,
            solver="lbfgs",
            **params,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_seed,
            **params,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=random_seed,
            **params,
        )
    raise ValueError(f"Unsupported model '{name}'. Expected one of: {', '.join(SUPPORTED_MODELS)}")


def model_requires_scaling(name: str) -> bool:
    """Return whether the estimator benefits from standardised numeric inputs."""

    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {name}")
    return name == "logistic_regression"
