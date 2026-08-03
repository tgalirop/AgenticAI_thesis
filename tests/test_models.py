"""Tests for estimator creation and fixed preprocessing."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from agenticai_thesis.modeling.models import create_estimator, model_requires_scaling


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("logistic_regression", LogisticRegression),
        ("decision_tree", DecisionTreeClassifier),
        ("random_forest", RandomForestClassifier),
    ],
)
def test_supported_estimators_are_created(name: str, expected_type: type) -> None:
    estimator = create_estimator(name, random_seed=42)
    assert isinstance(estimator, expected_type)
    assert estimator.random_state == 42


def test_only_logistic_regression_requires_scaling() -> None:
    assert model_requires_scaling("logistic_regression") is True
    assert model_requires_scaling("decision_tree") is False
    assert model_requires_scaling("random_forest") is False

