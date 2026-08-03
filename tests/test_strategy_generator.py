"""Tests for provider-neutral structured strategy generation."""

import pytest

from agenticai_thesis.agentic.domain import ColumnKind, DatasetContext, DatasetRole
from agenticai_thesis.agentic.model_clients import FakeModelClient, ModelClientError
from agenticai_thesis.agentic.prompts import StrategyPromptContext, StrategyPromptProvider
from agenticai_thesis.agentic.strategy_generator import (
    StrategyGenerator,
    TransformationPlanParseError,
    TransformationPlanParser,
)


def _context(iteration: int = 1) -> StrategyPromptContext:
    return StrategyPromptContext(
        dataset=DatasetContext(
            dataset_id="paysim-development-v1",
            role=DatasetRole.DEVELOPMENT,
            column_types={
                "amount": ColumnKind.NUMERIC,
                "type": ColumnKind.CATEGORICAL,
                "isFraud": ColumnKind.TARGET,
            },
            target_column="isFraud",
            protected_columns=frozenset({"isFraud"}),
        ),
        iteration=iteration,
        quality_metrics={"missing_rate": 0.01},
        baseline_metrics={"pr_auc": 0.83},
        previous_feedback=("Prefer a less expensive strategy.",),
    )


def _valid_response() -> dict[str, object]:
    return {
        "plan_id": "plan_001",
        "dataset_id": "paysim-development-v1",
        "iteration": 1,
        "objective": "Improve robustness to extreme amounts.",
        "actions": [
            {
                "action_id": "scale_1",
                "transformation": "scale_numeric",
                "columns": ["amount"],
                "parameters": {"method": "robust"},
                "rationale": "Reduce the effect of extreme transaction amounts.",
                "model_scope": "logistic_regression",
            }
        ],
    }


def _generator(fake: FakeModelClient) -> StrategyGenerator:
    return StrategyGenerator(
        model_client=fake,
        prompt_provider=StrategyPromptProvider(),
        parser=TransformationPlanParser(),
        allowed_transformations=("scale_numeric", "class_weight"),
    )


def test_generator_returns_typed_plan_and_sends_json_schema() -> None:
    fake = FakeModelClient([_valid_response()])
    plan = _generator(fake).generate(_context())

    assert plan.plan_id == "plan_001"
    assert plan.actions[0].transformation == "scale_numeric"
    assert fake.calls[0]["json_schema"]["additionalProperties"] is False
    assert "execute_python" not in fake.calls[0]["user_prompt"]
    assert "6.5" not in fake.calls[0]["user_prompt"]
    assert '"method": [' in fake.calls[0]["user_prompt"]
    assert '"robust"' in fake.calls[0]["user_prompt"]


def test_generator_rejects_extra_model_fields() -> None:
    response = _valid_response()
    response["python_code"] = "import os"
    with pytest.raises(TransformationPlanParseError, match="not a valid"):
        _generator(FakeModelClient([response])).generate(_context())


def test_generator_rejects_wrong_iteration() -> None:
    response = _valid_response()
    response["iteration"] = 2
    with pytest.raises(TransformationPlanParseError, match="unexpected iteration"):
        _generator(FakeModelClient([response])).generate(_context())


def test_fake_client_fails_clearly_when_queue_is_empty() -> None:
    with pytest.raises(ModelClientError, match="no queued response"):
        _generator(FakeModelClient([])).generate(_context())
