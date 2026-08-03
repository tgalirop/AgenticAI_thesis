"""Tests for immutable Agent state transitions and JSON checkpointing."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agenticai_thesis.agentic.checkpointing import (
    CheckpointStoreProtocol,
    JsonCheckpointStore,
)
from agenticai_thesis.agentic.domain import TransformationPlan, ValidationResult
from agenticai_thesis.agentic.feedback import (
    CandidateAssessment,
    FeedbackAction,
    FeedbackDecision,
    ModelOutcome,
)
from agenticai_thesis.agentic.state import (
    AgentRunStatus,
    AgentState,
    AgentStateManager,
    ArtifactReference,
    IterationRecord,
)


def _assessment(plan_id: str, iteration: int, score: float) -> CandidateAssessment:
    return CandidateAssessment(
        plan_id=plan_id,
        iteration=iteration,
        model_outcomes=(
            ModelOutcome(
                model_name="random_forest",
                status="success",
                primary_metric=score,
                recall=0.8,
                precision=0.7,
                fit_seconds=1.0,
            ),
        ),
        data_quality_score=0.95,
    )


def _record(iteration: int, action: FeedbackAction, *, new_best: bool = False) -> IterationRecord:
    plan = TransformationPlan(
        plan_id=f"plan_{iteration}",
        dataset_id="development-v1",
        iteration=iteration,
        objective="Test state transition.",
    )
    return IterationRecord(
        iteration=iteration,
        plan=plan,
        validation=ValidationResult(is_valid=True, plan=plan),
        assessment=_assessment(plan.plan_id, iteration, 0.70 + iteration / 100),
        feedback=FeedbackDecision(
            action=action,
            reason_code="test_decision",
            reason="Deterministic state test.",
            plan_id=plan.plan_id,
            iteration=iteration,
            is_new_best=new_best,
        ),
        artifacts=(
            ArtifactReference(
                artifact_type="transformation_plan",
                path=f"reports/transformation_plans/{plan.plan_id}.json",
            ),
        ),
    )


class FixedClock:
    """Injectable clock that advances one second per state transition."""

    def __init__(self) -> None:
        self.value = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        current = self.value
        self.value += timedelta(seconds=1)
        return current


def _state_manager_and_state() -> tuple[AgentStateManager, AgentState]:
    manager = AgentStateManager(clock=FixedClock())
    state = manager.create(
        run_id="run_001",
        dataset_id="development-v1",
        baseline=_assessment("conventional", 0, 0.70),
        max_iterations=3,
        initial_data_quality_score=0.95,
    )
    return manager, state


def test_state_manager_appends_retry_then_accepts_best_plan() -> None:
    manager, state = _state_manager_and_state()
    after_retry = manager.append_iteration(state, _record(1, FeedbackAction.RETRY))
    completed = manager.append_iteration(
        after_retry, _record(2, FeedbackAction.ACCEPT, new_best=True)
    )

    assert state.current_iteration == 0  # Immutable original state remains unchanged.
    assert after_retry.status == AgentRunStatus.RUNNING
    assert completed.status == AgentRunStatus.COMPLETED
    assert completed.current_iteration == 2
    assert completed.best_plan_id == "plan_2"
    assert completed.best_iteration == 2
    assert completed.best_primary_metric == pytest.approx(0.72)
    assert completed.termination_action == FeedbackAction.ACCEPT
    assert len(completed.history) == 2


def test_state_rejects_non_sequential_iteration() -> None:
    manager, state = _state_manager_and_state()
    with pytest.raises(ValueError, match="Expected iteration 1"):
        manager.append_iteration(state, _record(2, FeedbackAction.RETRY))


def test_completed_state_cannot_be_modified() -> None:
    manager, state = _state_manager_and_state()
    completed = manager.append_iteration(state, _record(1, FeedbackAction.ACCEPT))
    with pytest.raises(ValueError, match="completed"):
        manager.append_iteration(completed, _record(2, FeedbackAction.RETRY))


def test_json_checkpoint_round_trip_is_lossless(tmp_path: Path) -> None:
    manager, state = _state_manager_and_state()
    state = manager.append_iteration(state, _record(1, FeedbackAction.RETRY))
    store = JsonCheckpointStore(tmp_path / "checkpoints")

    assert isinstance(store, CheckpointStoreProtocol)
    path = store.save(state)
    loaded = store.load("run_001")

    assert path == (tmp_path / "checkpoints" / "run_001.json").resolve()
    assert store.exists("run_001") is True
    assert loaded == state
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("unsafe_id", ["../escape", "folder/run", "run.json", ""])
def test_checkpoint_store_rejects_unsafe_run_ids(tmp_path: Path, unsafe_id: str) -> None:
    store = JsonCheckpointStore(tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        store.exists(unsafe_id)
