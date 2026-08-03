"""Tests for the object-oriented Machine Learning Evaluator."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from agenticai_thesis.agentic.evaluators import EvaluationConfig, MachineLearningEvaluator
from agenticai_thesis.modeling.cross_validation import CrossValidationFoldProvider


def _classification_data() -> tuple[pd.DataFrame, np.ndarray]:
    x, y = make_classification(
        n_samples=90,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.75, 0.25],
        random_state=42,
    )
    return pd.DataFrame(x, columns=["a", "b", "c", "d"]), y


def test_evaluator_produces_typed_fold_results_and_oof_predictions() -> None:
    x, y = _classification_data()
    folds = CrossValidationFoldProvider(folds=3, repeats=2, random_seed=42).create(y)
    pipeline = LogisticRegression(max_iter=300, random_state=42)
    output = MachineLearningEvaluator(EvaluationConfig()).evaluate(
        pipeline, x, y, folds, model_name="logistic_regression"
    )

    assert output.result.status == "success"
    assert output.result.successful_folds == 6
    assert output.result.failed_folds == 0
    assert output.result.primary_metric == "pr_auc"
    assert output.result.primary_metric_mean is not None
    assert 0.0 <= output.result.primary_metric_mean <= 1.0
    assert len(output.predictions) == len(y) * 2
    assert set(output.result.metric_summary) == {
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
    }
    # The evaluator fits clones, leaving the reusable input pipeline untouched.
    assert not hasattr(pipeline, "coef_")


def test_evaluator_is_deterministic_with_same_folds_and_estimator_seed() -> None:
    x, y = _classification_data()
    folds = CrossValidationFoldProvider(folds=3, repeats=1, random_seed=42).create(y)
    evaluator = MachineLearningEvaluator(EvaluationConfig())
    outputs = [
        evaluator.evaluate(
            LogisticRegression(max_iter=300, random_state=42),
            x,
            y,
            folds,
            model_name="logistic_regression",
        )
        for _ in range(2)
    ]
    np.testing.assert_allclose(outputs[0].predictions["y_score"], outputs[1].predictions["y_score"])
    assert outputs[0].result.primary_metric_mean == outputs[1].result.primary_metric_mean


def test_evaluator_rejects_fold_set_for_different_target() -> None:
    x, y = _classification_data()
    folds = CrossValidationFoldProvider(folds=3, repeats=1, random_seed=42).create(y)
    changed = y.copy()
    changed[0] = 1 - changed[0]
    with pytest.raises(ValueError, match="fingerprint"):
        MachineLearningEvaluator(EvaluationConfig()).evaluate(
            LogisticRegression(), x, changed, folds, model_name="logistic_regression"
        )


class AlwaysFailClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable test estimator used to verify typed execution-error feedback."""

    def fit(self, x: object, y: object) -> "AlwaysFailClassifier":
        raise RuntimeError("deliberate training failure")

    def predict_proba(self, x: object) -> np.ndarray:  # pragma: no cover - fit always fails.
        raise AssertionError("predict_proba must not be called")


def test_evaluator_records_all_fold_failures_without_raw_exception() -> None:
    x, y = _classification_data()
    folds = CrossValidationFoldProvider(folds=3, repeats=1, random_seed=42).create(y)
    output = MachineLearningEvaluator(EvaluationConfig()).evaluate(
        AlwaysFailClassifier(), x, y, folds, model_name="failing_model"
    )

    assert output.result.status == "error"
    assert output.result.successful_folds == 0
    assert output.result.failed_folds == 3
    assert output.result.primary_metric_mean is None
    assert output.predictions.empty
    assert all(fold.error_type == "RuntimeError" for fold in output.result.fold_results)
    assert all("deliberate training failure" in fold.error_message for fold in output.result.fold_results)
