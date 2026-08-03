"""Tests for deterministic benefit/cost feedback and stopping decisions."""

import pytest

from agenticai_thesis.agentic.feedback import (
    CandidateAssessment,
    FeedbackAction,
    FeedbackPolicy,
    FeedbackPolicyConfig,
    ModelOutcome,
)


def _outcome(
    name: str,
    primary: float,
    *,
    recall: float = 0.80,
    precision: float = 0.70,
    status: str = "success",
    seconds: float = 1.0,
) -> ModelOutcome:
    return ModelOutcome(
        model_name=name,
        status=status,
        primary_metric=primary if status == "success" else None,
        recall=recall if status == "success" else None,
        precision=precision if status == "success" else None,
        fit_seconds=seconds,
    )


def _assessment(
    plan_id: str,
    iteration: int,
    primary: float = 0.70,
    *,
    recall: float = 0.80,
    precision: float = 0.70,
    quality: float = 0.90,
    validation_valid: bool = True,
    execution_status: str = "success",
    seconds: float = 1.0,
    model_status: str = "success",
) -> CandidateAssessment:
    return CandidateAssessment(
        plan_id=plan_id,
        iteration=iteration,
        validation_valid=validation_valid,
        execution_status=execution_status,
        model_outcomes=(
            _outcome(
                "logistic_regression",
                primary,
                recall=recall,
                precision=precision,
                status=model_status,
                seconds=seconds,
            ),
            _outcome(
                "random_forest",
                primary,
                recall=recall,
                precision=precision,
                status=model_status,
                seconds=seconds,
            ),
        ),
        data_quality_score=quality,
    )


def _baseline() -> CandidateAssessment:
    return _assessment("conventional", 0, 0.70, quality=0.90)


def _policy(**overrides: object) -> FeedbackPolicy:
    values = {
        "max_iterations": 3,
        "no_improvement_patience": 2,
        "max_invalid_strategies": 2,
        "max_execution_errors": 2,
        "minimum_primary_improvement": 0.005,
        "minimum_quality_improvement": 0.0001,
        "maximum_recall_drop": 0.02,
        "maximum_precision_drop": 0.02,
        "maximum_runtime_multiplier": 3.0,
        "require_all_models_successful": True,
    }
    values.update(overrides)
    return FeedbackPolicy(FeedbackPolicyConfig.model_validate(values))


def test_policy_accepts_material_improvement_and_marks_new_best() -> None:
    decision = _policy().decide(_assessment("plan_1", 1, 0.72), _baseline())
    assert decision.action == FeedbackAction.ACCEPT
    assert decision.primary_metric_delta == pytest.approx(0.02)
    assert decision.is_new_best is True


def test_policy_accepts_quality_gain_at_equal_performance_as_selected_best() -> None:
    decision = _policy().decide(
        _assessment("plan_quality", 1, 0.70, quality=0.91), _baseline()
    )
    assert decision.action == FeedbackAction.ACCEPT
    assert decision.primary_metric_delta == pytest.approx(0.0)
    assert decision.quality_delta == pytest.approx(0.01)
    assert decision.is_new_best is True


def test_policy_retries_when_recall_guardrail_is_violated() -> None:
    decision = _policy().decide(
        _assessment("plan_1", 1, 0.75, recall=0.70), _baseline()
    )
    assert decision.action == FeedbackAction.RETRY
    assert decision.reason_code == "metric_guardrail_violation"
    assert any("Recall" in recommendation for recommendation in decision.recommendations)


def test_policy_retries_excessive_runtime_despite_metric_gain() -> None:
    decision = _policy().decide(
        _assessment("plan_1", 1, 0.75, seconds=5.0), _baseline()
    )
    assert decision.action == FeedbackAction.RETRY
    assert decision.reason_code == "excessive_runtime_cost"
    assert decision.runtime_multiplier == pytest.approx(5.0)


def test_policy_stops_after_two_non_improving_iterations() -> None:
    first = _assessment("plan_1", 1, 0.701)
    second = _assessment("plan_2", 2, 0.702)
    decision = _policy().decide(second, _baseline(), history=(first,))
    assert decision.action == FeedbackAction.STOP_NO_IMPROVEMENT


def test_policy_stops_at_maximum_iteration() -> None:
    decision = _policy().decide(_assessment("plan_3", 3, 0.701), _baseline())
    assert decision.action == FeedbackAction.STOP_MAX_ITERATIONS


def test_policy_stops_after_repeated_invalid_strategies() -> None:
    first = _assessment("invalid_1", 1, validation_valid=False)
    second = _assessment("invalid_2", 2, validation_valid=False)
    decision = _policy().decide(second, _baseline(), history=(first,))
    assert decision.action == FeedbackAction.STOP_INVALID_STRATEGIES


def test_policy_stops_immediately_on_fatal_execution_error() -> None:
    candidate = _assessment("fatal_1", 1, execution_status="fatal_error")
    decision = _policy().decide(candidate, _baseline())
    assert decision.action == FeedbackAction.STOP_EXECUTION_ERROR


def test_policy_rejects_partial_model_evaluation() -> None:
    candidate = _assessment("partial_1", 1, model_status="partial_failure")
    decision = _policy().decide(candidate, _baseline())
    assert decision.action == FeedbackAction.RETRY
    assert decision.reason_code == "model_evaluation_failure"


def test_policy_requires_same_models_as_baseline() -> None:
    candidate = CandidateAssessment(
        plan_id="plan_1",
        iteration=1,
        model_outcomes=(_outcome("decision_tree", 0.75),),
        data_quality_score=0.9,
    )
    with pytest.raises(ValueError, match="same model names"):
        _policy().decide(candidate, _baseline())
