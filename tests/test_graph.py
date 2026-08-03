"""End-to-end tests for LangGraph orchestration with deterministic fakes."""

from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agenticai_thesis.agentic.checkpointing import JsonCheckpointStore
from agenticai_thesis.agentic.domain import ColumnKind, DatasetContext, DatasetRole
from agenticai_thesis.agentic.feedback import (
    CandidateAssessment,
    FeedbackAction,
    FeedbackPolicy,
    FeedbackPolicyConfig,
    ModelOutcome,
)
from agenticai_thesis.agentic.graph import (
    AgentGraphDependencies,
    AgentWorkflow,
    CandidateEvaluationError,
    CandidateEvaluatorProtocol,
)
from agenticai_thesis.agentic.model_clients import FakeModelClient
from agenticai_thesis.agentic.prompts import StrategyPromptProvider
from agenticai_thesis.agentic.state import AgentRunStatus, AgentStateManager
from agenticai_thesis.agentic.strategy_generator import (
    StrategyGenerator,
    TransformationPlanParser,
)
from agenticai_thesis.agentic.transformations import TransformationRegistry
from agenticai_thesis.agentic.validator import TransformationPlanValidator


def _model_outcome(score: float) -> ModelOutcome:
    return ModelOutcome(
        model_name="random_forest",
        status="success",
        primary_metric=score,
        recall=0.80,
        precision=0.70,
        fit_seconds=0.1,
        predict_seconds=0.01,
    )


def _assessment(plan_id: str, iteration: int, score: float) -> CandidateAssessment:
    return CandidateAssessment(
        plan_id=plan_id,
        iteration=iteration,
        model_outcomes=(_model_outcome(score),),
        data_quality_score=0.90,
        preprocessing_seconds=0.01,
    )


class QueuedCandidateEvaluator:
    """Return predefined evidence while recording validated-plan calls."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = deque(scores)
        self.plan_ids: list[str] = []

    def evaluate(self, *, validation: object, dataset_context: DatasetContext) -> CandidateAssessment:
        # Runtime attributes are intentionally accessed only after the graph's
        # validator routed the plan to this service.
        plan = validation.plan  # type: ignore[attr-defined]
        assert plan is not None
        assert dataset_context.role == DatasetRole.DEVELOPMENT
        self.plan_ids.append(plan.plan_id)
        return _assessment(plan.plan_id, plan.iteration, self._scores.popleft())


class FailingCandidateEvaluator:
    """Raise an expected recoverable error on every evaluation attempt."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, *, validation: object, dataset_context: DatasetContext) -> CandidateAssessment:
        self.calls += 1
        raise CandidateEvaluationError("deliberate recoverable execution failure")


def _plan(plan_id: str, iteration: int, transformation: str = "class_weight") -> dict:
    parameters = {"mode": "balanced"} if transformation == "class_weight" else {}
    return {
        "plan_id": plan_id,
        "dataset_id": "development-v1",
        "iteration": iteration,
        "objective": "Improve preprocessing safely.",
        "actions": [
            {
                "action_id": f"action_{iteration}",
                "transformation": transformation,
                "columns": [],
                "parameters": parameters,
                "rationale": "Use a deterministic allowlisted operation.",
                "model_scope": "all",
            }
        ],
    }


def _context(role: DatasetRole = DatasetRole.DEVELOPMENT) -> DatasetContext:
    return DatasetContext(
        dataset_id="development-v1",
        role=role,
        column_types={"amount": ColumnKind.NUMERIC, "isFraud": ColumnKind.TARGET},
        target_column="isFraud",
        protected_columns=frozenset({"isFraud"}),
    )


def _workflow(
    tmp_path: Path,
    responses: list[dict],
    evaluator: CandidateEvaluatorProtocol,
) -> tuple[AgentWorkflow, FakeModelClient, JsonCheckpointStore]:
    model = FakeModelClient(responses)
    registry = TransformationRegistry.default()
    generator = StrategyGenerator(
        model_client=model,
        prompt_provider=StrategyPromptProvider(),
        parser=TransformationPlanParser(),
        allowed_transformations=tuple(registry.names),
    )
    store = JsonCheckpointStore(tmp_path / "checkpoints")
    dependencies = AgentGraphDependencies(
        strategy_generator=generator,
        validator=TransformationPlanValidator(registry),
        candidate_evaluator=evaluator,
        feedback_policy=FeedbackPolicy(
            FeedbackPolicyConfig(
                max_iterations=3,
                no_improvement_patience=2,
                max_invalid_strategies=2,
                max_execution_errors=2,
            )
        ),
        state_manager=AgentStateManager(clock=lambda: datetime(2026, 8, 3, tzinfo=UTC)),
        checkpoint_store=store,
    )
    assert isinstance(evaluator, CandidateEvaluatorProtocol)
    return AgentWorkflow(dependencies), model, store


def _initial_state() -> object:
    return AgentStateManager(clock=lambda: datetime(2026, 8, 3, tzinfo=UTC)).create(
        run_id="graph_run_001",
        dataset_id="development-v1",
        baseline=_assessment("conventional", 0, 0.70),
        max_iterations=3,
        initial_data_quality_score=0.90,
    )


def test_graph_accepts_improving_plan_and_checkpoints_state(tmp_path: Path) -> None:
    evaluator = QueuedCandidateEvaluator([0.72])
    workflow, model, store = _workflow(tmp_path, [_plan("plan_001", 1)], evaluator)

    result = workflow.run(
        agent_state=_initial_state(),  # type: ignore[arg-type]
        dataset_context=_context(),
        quality_metrics={"data_quality_score": 0.90},
        baseline_metrics={"pr_auc": 0.70},
    )

    assert result.status == AgentRunStatus.COMPLETED
    assert result.termination_action == FeedbackAction.ACCEPT
    assert result.best_plan_id == "plan_001"
    assert len(result.history) == 1
    assert evaluator.plan_ids == ["plan_001"]
    assert len(model.calls) == 1
    assert store.load("graph_run_001") == result


def test_invalid_plan_is_not_executed_and_retry_can_succeed(tmp_path: Path) -> None:
    evaluator = QueuedCandidateEvaluator([0.73])
    responses = [_plan("invalid_001", 1, "execute_python"), _plan("plan_002", 2)]
    workflow, model, _store = _workflow(tmp_path, responses, evaluator)

    result = workflow.run(
        agent_state=_initial_state(),  # type: ignore[arg-type]
        dataset_context=_context(),
        quality_metrics={"data_quality_score": 0.90},
        baseline_metrics={"pr_auc": 0.70},
    )

    assert result.termination_action == FeedbackAction.ACCEPT
    assert len(result.history) == 2
    assert result.history[0].validation.is_valid is False
    assert result.history[0].feedback.action == FeedbackAction.RETRY
    assert "transformation_not_allowed" in result.history[0].warnings[0]
    assert evaluator.plan_ids == ["plan_002"]
    assert len(model.calls) == 2


def test_temporal_test_is_rejected_before_calling_model(tmp_path: Path) -> None:
    evaluator = QueuedCandidateEvaluator([0.72])
    workflow, model, _store = _workflow(tmp_path, [_plan("plan_001", 1)], evaluator)

    with pytest.raises(ValueError, match="development data"):
        workflow.run(
            agent_state=_initial_state(),  # type: ignore[arg-type]
            dataset_context=_context(DatasetRole.TEMPORAL_TEST),
            quality_metrics={},
            baseline_metrics={},
        )

    assert model.calls == []
    assert evaluator.plan_ids == []


def test_repeated_execution_errors_stop_and_remain_auditable(tmp_path: Path) -> None:
    evaluator = FailingCandidateEvaluator()
    responses = [_plan("plan_001", 1), _plan("plan_002", 2)]
    workflow, model, store = _workflow(tmp_path, responses, evaluator)

    result = workflow.run(
        agent_state=_initial_state(),  # type: ignore[arg-type]
        dataset_context=_context(),
        quality_metrics={"data_quality_score": 0.90},
        baseline_metrics={"pr_auc": 0.70},
    )

    assert result.termination_action == FeedbackAction.STOP_EXECUTION_ERROR
    assert evaluator.calls == 2
    assert len(model.calls) == 2
    assert all(record.assessment.execution_status == "recoverable_error" for record in result.history)
    assert "deliberate recoverable execution failure" in result.history[-1].errors[0]
    assert store.load("graph_run_001") == result
