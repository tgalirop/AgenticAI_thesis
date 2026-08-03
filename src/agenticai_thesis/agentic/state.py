"""Immutable, serializable Agent state and controlled state transitions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from agenticai_thesis.agentic.domain import TransformationPlan, ValidationResult
from agenticai_thesis.agentic.feedback import (
    CandidateAssessment,
    FeedbackAction,
    FeedbackDecision,
)


class AgentRunStatus(StrEnum):
    """Lifecycle state of one complete Agent run."""

    RUNNING = "running"
    COMPLETED = "completed"


class ArtifactReference(BaseModel):
    """Serializable pointer to an artifact kept outside compact Agent state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_type: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class IterationRecord(BaseModel):
    """Complete serializable audit record for one Agent iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration: int = Field(ge=1)
    plan: TransformationPlan
    validation: ValidationResult
    assessment: CandidateAssessment
    feedback: FeedbackDecision
    artifacts: tuple[ArtifactReference, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class AgentState(BaseModel):
    """LangGraph-compatible state containing no fitted Python objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    dataset_id: str = Field(min_length=1)
    status: AgentRunStatus = AgentRunStatus.RUNNING
    current_iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=3, ge=1)
    baseline: CandidateAssessment
    initial_data_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    history: tuple[IterationRecord, ...] = ()
    best_plan_id: str | None = None
    best_iteration: int | None = None
    best_primary_metric: float | None = Field(default=None, ge=0.0, le=1.0)
    termination_action: FeedbackAction | None = None
    created_at_utc: datetime
    updated_at_utc: datetime


class AgentStateManager:
    """Only service allowed to create and advance immutable AgentState objects."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    def create(
        self,
        *,
        run_id: str,
        dataset_id: str,
        baseline: CandidateAssessment,
        max_iterations: int,
        initial_data_quality_score: float | None,
    ) -> AgentState:
        """Create a new empty run using one injected baseline assessment."""

        if baseline.iteration != 0:
            raise ValueError("Baseline assessment must use reserved iteration zero")
        now = self._utc(self._clock())
        return AgentState(
            run_id=run_id,
            dataset_id=dataset_id,
            max_iterations=max_iterations,
            baseline=baseline,
            initial_data_quality_score=initial_data_quality_score,
            created_at_utc=now,
            updated_at_utc=now,
        )

    def append_iteration(self, state: AgentState, record: IterationRecord) -> AgentState:
        """Validate transition invariants and return a new immutable state."""

        if state.status != AgentRunStatus.RUNNING:
            raise ValueError("Cannot append an iteration to a completed Agent run")
        expected_iteration = state.current_iteration + 1
        if record.iteration != expected_iteration:
            raise ValueError(f"Expected iteration {expected_iteration}, received {record.iteration}")
        if record.iteration > state.max_iterations:
            raise ValueError("Iteration exceeds the Agent run maximum")
        if record.plan.dataset_id != state.dataset_id:
            raise ValueError("Iteration plan refers to a different dataset")
        if record.plan.iteration != record.iteration:
            raise ValueError("Plan iteration does not match IterationRecord")
        if record.assessment.iteration != record.iteration:
            raise ValueError("Assessment iteration does not match IterationRecord")
        if record.feedback.iteration != record.iteration:
            raise ValueError("Feedback iteration does not match IterationRecord")
        if record.feedback.plan_id != record.plan.plan_id:
            raise ValueError("Feedback plan_id does not match the recorded plan")

        best_plan_id = state.best_plan_id
        best_iteration = state.best_iteration
        best_primary_metric = state.best_primary_metric
        if record.feedback.is_new_best:
            score = record.assessment.mean_primary_metric
            if score is None:
                raise ValueError("A new-best decision requires a candidate primary metric")
            best_plan_id = record.plan.plan_id
            best_iteration = record.iteration
            best_primary_metric = score

        terminal = record.feedback.action != FeedbackAction.RETRY
        return state.model_copy(
            update={
                "status": AgentRunStatus.COMPLETED if terminal else AgentRunStatus.RUNNING,
                "current_iteration": record.iteration,
                "history": (*state.history, record),
                "best_plan_id": best_plan_id,
                "best_iteration": best_iteration,
                "best_primary_metric": best_primary_metric,
                "termination_action": record.feedback.action if terminal else None,
                "updated_at_utc": self._utc(self._clock()),
            }
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        """Reject ambiguous naive timestamps and normalize aware values to UTC."""

        if value.tzinfo is None:
            raise ValueError("AgentState timestamps must be timezone-aware")
        return value.astimezone(UTC)
