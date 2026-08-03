"""Deterministic feedback and termination policy for Agent iterations."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from statistics import mean
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeedbackAction(StrEnum):
    """All transitions the feedback node may return to LangGraph."""

    ACCEPT = "ACCEPT"
    RETRY = "RETRY"
    STOP_NO_IMPROVEMENT = "STOP_NO_IMPROVEMENT"
    STOP_MAX_ITERATIONS = "STOP_MAX_ITERATIONS"
    STOP_INVALID_STRATEGIES = "STOP_INVALID_STRATEGIES"
    STOP_EXECUTION_ERROR = "STOP_EXECUTION_ERROR"


class ModelOutcome(BaseModel):
    """Compact comparable outcome for one model in one strategy evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str = Field(min_length=1)
    status: Literal["success", "partial_failure", "error"]
    primary_metric: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    fit_seconds: float = Field(default=0.0, ge=0.0)
    predict_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def successful_outcome_requires_metrics(self) -> "ModelOutcome":
        if self.status == "success" and any(
            value is None for value in (self.primary_metric, self.recall, self.precision)
        ):
            raise ValueError("Successful model outcomes require all comparison metrics")
        return self


class CandidateAssessment(BaseModel):
    """Serializable benefit/cost evidence produced for one transformation plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1)
    # Iteration zero is reserved for the conventional baseline reference.
    iteration: int = Field(ge=0)
    validation_valid: bool = True
    execution_status: Literal["success", "recoverable_error", "fatal_error"] = "success"
    model_outcomes: tuple[ModelOutcome, ...] = ()
    data_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    preprocessing_seconds: float = Field(default=0.0, ge=0.0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def model_names_must_be_unique(self) -> "CandidateAssessment":
        names = [outcome.model_name for outcome in self.model_outcomes]
        if len(names) != len(set(names)):
            raise ValueError("Candidate model outcomes must have unique model names")
        return self

    @property
    def successful_outcomes(self) -> tuple[ModelOutcome, ...]:
        return tuple(outcome for outcome in self.model_outcomes if outcome.status == "success")

    @property
    def mean_primary_metric(self) -> float | None:
        values = [outcome.primary_metric for outcome in self.successful_outcomes]
        return mean(value for value in values if value is not None) if values else None

    @property
    def total_runtime_seconds(self) -> float:
        return self.preprocessing_seconds + sum(
            outcome.fit_seconds + outcome.predict_seconds for outcome in self.model_outcomes
        )


class FeedbackPolicyConfig(BaseModel):
    """Immutable thresholds loaded from configuration, never hidden in policy code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_iterations: int = Field(default=3, ge=1)
    no_improvement_patience: int = Field(default=2, ge=1)
    max_invalid_strategies: int = Field(default=2, ge=1)
    max_execution_errors: int = Field(default=2, ge=1)
    minimum_primary_improvement: float = Field(default=0.005, ge=0.0)
    minimum_quality_improvement: float = Field(default=0.0001, ge=0.0)
    maximum_recall_drop: float = Field(default=0.02, ge=0.0, le=1.0)
    maximum_precision_drop: float = Field(default=0.02, ge=0.0, le=1.0)
    maximum_runtime_multiplier: float = Field(default=3.0, ge=1.0)
    require_all_models_successful: bool = True


class FeedbackDecision(BaseModel):
    """Typed routing decision stored in Agent state and consumed by LangGraph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: FeedbackAction
    reason_code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    plan_id: str
    iteration: int
    primary_metric_delta: float | None = None
    quality_delta: float | None = None
    runtime_multiplier: float | None = None
    is_new_best: bool = False
    recommendations: tuple[str, ...] = ()


class FeedbackPolicy:
    """Apply transparent benefit, guardrail, failure, and stopping rules."""

    def __init__(self, config: FeedbackPolicyConfig) -> None:
        self._config = config

    def decide(
        self,
        candidate: CandidateAssessment,
        baseline: CandidateAssessment,
        history: tuple[CandidateAssessment, ...] = (),
    ) -> FeedbackDecision:
        """Compare one candidate with baseline and previous Agent attempts."""

        if not candidate.validation_valid:
            invalid_count = self._consecutive_count(
                (*history, candidate), lambda item: not item.validation_valid
            )
            action = (
                FeedbackAction.STOP_INVALID_STRATEGIES
                if invalid_count >= self._config.max_invalid_strategies
                else FeedbackAction.RETRY
            )
            return self._decision(
                candidate,
                action,
                "invalid_strategy",
                "The transformation plan did not pass deterministic validation.",
                recommendations=("Correct the validation issues before generating another plan.",),
            )

        if candidate.execution_status != "success":
            error_count = self._consecutive_count(
                (*history, candidate), lambda item: item.execution_status != "success"
            )
            must_stop = (
                candidate.execution_status == "fatal_error"
                or error_count >= self._config.max_execution_errors
            )
            return self._decision(
                candidate,
                FeedbackAction.STOP_EXECUTION_ERROR if must_stop else FeedbackAction.RETRY,
                "execution_error",
                "The validated strategy could not be executed successfully.",
                recommendations=("Revise the actions or parameters reported by the Executor.",),
            )

        candidate_models = {outcome.model_name: outcome for outcome in candidate.model_outcomes}
        baseline_models = {outcome.model_name: outcome for outcome in baseline.model_outcomes}
        if set(candidate_models) != set(baseline_models):
            raise ValueError("Candidate and baseline must contain exactly the same model names")
        if self._config.require_all_models_successful and any(
            outcome.status != "success" for outcome in candidate.model_outcomes
        ):
            return self._retry_or_stop(
                candidate,
                history,
                reason_code="model_evaluation_failure",
                reason="At least one model did not complete all evaluation folds.",
                recommendations=("Revise the plan using the recorded fold errors.",),
            )

        candidate_primary = candidate.mean_primary_metric
        baseline_primary = baseline.mean_primary_metric
        if candidate_primary is None or baseline_primary is None:
            return self._retry_or_stop(
                candidate,
                history,
                reason_code="missing_metrics",
                reason="Comparable primary metrics are unavailable.",
                recommendations=("Resolve model evaluation failures before comparison.",),
            )

        primary_delta = candidate_primary - baseline_primary
        quality_delta = self._optional_delta(
            candidate.data_quality_score, baseline.data_quality_score
        )
        runtime_multiplier = self._runtime_multiplier(candidate, baseline)
        guardrail_messages: list[str] = []
        for model_name, outcome in candidate_models.items():
            reference = baseline_models[model_name]
            if outcome.status != "success" or reference.status != "success":
                continue
            if outcome.recall is not None and reference.recall is not None:
                if reference.recall - outcome.recall > self._config.maximum_recall_drop:
                    guardrail_messages.append(f"Restore Recall for {model_name}.")
            if outcome.precision is not None and reference.precision is not None:
                if reference.precision - outcome.precision > self._config.maximum_precision_drop:
                    guardrail_messages.append(f"Restore Precision for {model_name}.")

        if guardrail_messages:
            return self._retry_or_stop(
                candidate,
                history,
                reason_code="metric_guardrail_violation",
                reason="The candidate violates model-level Recall or Precision guardrails.",
                primary_delta=primary_delta,
                quality_delta=quality_delta,
                runtime_multiplier=runtime_multiplier,
                recommendations=tuple(guardrail_messages),
            )
        if (
            runtime_multiplier is not None
            and runtime_multiplier > self._config.maximum_runtime_multiplier
        ):
            return self._retry_or_stop(
                candidate,
                history,
                reason_code="excessive_runtime_cost",
                reason="The candidate exceeds the configured runtime cost limit.",
                primary_delta=primary_delta,
                quality_delta=quality_delta,
                runtime_multiplier=runtime_multiplier,
                recommendations=("Prefer a less expensive preprocessing strategy.",),
            )

        performance_improved = primary_delta >= self._config.minimum_primary_improvement
        quality_improved = (
            quality_delta is not None
            and quality_delta >= self._config.minimum_quality_improvement
            and primary_delta >= 0.0
        )
        if performance_improved or quality_improved:
            return self._decision(
                candidate,
                FeedbackAction.ACCEPT,
                "improvement_accepted",
                "The candidate provides an acceptable benefit without violating guardrails.",
                primary_delta=primary_delta,
                quality_delta=quality_delta,
                runtime_multiplier=runtime_multiplier,
                # ACCEPT terminates the search and selects this candidate. Mark it
                # as best even when acceptance comes from a quality improvement at
                # equal predictive performance.
                is_new_best=True,
            )
        return self._retry_or_stop(
            candidate,
            history,
            reason_code="insufficient_improvement",
            reason="The candidate does not reach the configured improvement threshold.",
            primary_delta=primary_delta,
            quality_delta=quality_delta,
            runtime_multiplier=runtime_multiplier,
            recommendations=("Generate a materially different preprocessing strategy.",),
        )

    def _retry_or_stop(
        self,
        candidate: CandidateAssessment,
        history: tuple[CandidateAssessment, ...],
        *,
        reason_code: str,
        reason: str,
        primary_delta: float | None = None,
        quality_delta: float | None = None,
        runtime_multiplier: float | None = None,
        recommendations: tuple[str, ...] = (),
    ) -> FeedbackDecision:
        if candidate.iteration >= self._config.max_iterations:
            action = FeedbackAction.STOP_MAX_ITERATIONS
        else:
            non_improving = self._consecutive_non_improving((*history, candidate))
            action = (
                FeedbackAction.STOP_NO_IMPROVEMENT
                if non_improving >= self._config.no_improvement_patience
                else FeedbackAction.RETRY
            )
        return self._decision(
            candidate,
            action,
            reason_code,
            reason,
            primary_delta=primary_delta,
            quality_delta=quality_delta,
            runtime_multiplier=runtime_multiplier,
            recommendations=recommendations,
        )

    def _consecutive_non_improving(
        self, assessments: tuple[CandidateAssessment, ...]
    ) -> int:
        count = 0
        for index in range(len(assessments) - 1, -1, -1):
            assessment = assessments[index]
            if not assessment.validation_valid or assessment.execution_status != "success":
                break
            # A prior assessment is considered non-improving when it did not clear
            # the baseline-independent minimum gain over the immediately preceding
            # successful score. The first attempt cannot establish a streak alone.
            if index == 0:
                count += 1
                continue
            previous = assessments[index - 1].mean_primary_metric
            current = assessment.mean_primary_metric
            if previous is None or current is None or current - previous < self._config.minimum_primary_improvement:
                count += 1
            else:
                break
        return count

    @staticmethod
    def _consecutive_count(
        assessments: tuple[CandidateAssessment, ...],
        predicate: Callable[[CandidateAssessment], bool],
    ) -> int:
        count = 0
        for assessment in reversed(assessments):
            if not predicate(assessment):
                break
            count += 1
        return count

    @staticmethod
    def _optional_delta(value: float | None, reference: float | None) -> float | None:
        return value - reference if value is not None and reference is not None else None

    @staticmethod
    def _runtime_multiplier(
        candidate: CandidateAssessment, baseline: CandidateAssessment
    ) -> float | None:
        baseline_runtime = baseline.total_runtime_seconds
        if baseline_runtime == 0.0:
            return None
        return candidate.total_runtime_seconds / baseline_runtime

    @staticmethod
    def _decision(
        candidate: CandidateAssessment,
        action: FeedbackAction,
        reason_code: str,
        reason: str,
        *,
        primary_delta: float | None = None,
        quality_delta: float | None = None,
        runtime_multiplier: float | None = None,
        is_new_best: bool = False,
        recommendations: tuple[str, ...] = (),
    ) -> FeedbackDecision:
        return FeedbackDecision(
            action=action,
            reason_code=reason_code,
            reason=reason,
            plan_id=candidate.plan_id,
            iteration=candidate.iteration,
            primary_metric_delta=primary_delta,
            quality_delta=quality_delta,
            runtime_multiplier=runtime_multiplier,
            is_new_best=is_new_best,
            recommendations=recommendations,
        )
