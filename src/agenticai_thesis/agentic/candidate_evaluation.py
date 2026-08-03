"""Production composition of preprocessing, quality, and ML evaluation.

Large feature matrices, target arrays, folds, pipelines, and estimators stay in
this service.  LangGraph receives only the compact :class:`CandidateAssessment`,
which keeps checkpoints serializable and prevents accidental raw-data leakage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from agenticai_thesis.agentic.domain import DatasetContext, DatasetRole, ValidationResult
from agenticai_thesis.agentic.evaluators import MachineLearningEvaluator, MetricAggregate
from agenticai_thesis.agentic.executor import TransformationExecutor
from agenticai_thesis.agentic.feedback import CandidateAssessment, ModelOutcome
from agenticai_thesis.agentic.graph import CandidateEvaluationError
from agenticai_thesis.modeling.cross_validation import CrossValidationFoldSet


@dataclass(frozen=True, slots=True)
class PlanQualityEvaluation:
    """Compact quality result for a safely executed transformation plan."""

    data_quality_score: float
    evaluation_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.data_quality_score <= 1.0:
            raise ValueError("data_quality_score must be between zero and one")
        if self.evaluation_seconds < 0.0:
            raise ValueError("evaluation_seconds cannot be negative")


@runtime_checkable
class PlanQualityEvaluatorProtocol(Protocol):
    """Measure quality after applying a plan to development/training data only."""

    def evaluate(
        self,
        *,
        validation: ValidationResult,
        dataset_context: DatasetContext,
    ) -> PlanQualityEvaluation:
        """Return a compact score without exposing transformed rows to the graph."""


class AgentCandidateEvaluator:
    """Evaluate the same valid plan across all configured fraud models."""

    def __init__(
        self,
        *,
        executor: TransformationExecutor,
        quality_evaluator: PlanQualityEvaluatorProtocol,
        ml_evaluator: MachineLearningEvaluator,
        estimators: Mapping[str, ClassifierMixin],
        features: pd.DataFrame,
        target: np.ndarray,
        folds: CrossValidationFoldSet,
    ) -> None:
        if not isinstance(quality_evaluator, PlanQualityEvaluatorProtocol):
            raise TypeError("quality_evaluator must satisfy PlanQualityEvaluatorProtocol")
        if not estimators:
            raise ValueError("At least one estimator is required")
        if len(features) != len(target):
            raise ValueError("Feature and target row counts do not match")
        folds.validate_target(target)

        self._executor = executor
        self._quality_evaluator = quality_evaluator
        self._ml_evaluator = ml_evaluator
        self._estimators = dict(estimators)
        self._features = features
        self._target = np.asarray(target, dtype=int)
        self._folds = folds

    def evaluate(
        self,
        *,
        validation: ValidationResult,
        dataset_context: DatasetContext,
    ) -> CandidateAssessment:
        """Build fold-local pipelines and return comparable model evidence."""

        if not validation.is_valid or validation.plan is None:
            raise ValueError("AgentCandidateEvaluator requires a valid plan")
        if dataset_context.role != DatasetRole.DEVELOPMENT:
            raise ValueError("Candidate evaluation is permitted only on development data")

        plan = validation.plan
        model_outcomes: list[ModelOutcome] = []
        pipeline_build_seconds = 0.0
        try:
            quality = self._quality_evaluator.evaluate(
                validation=validation,
                dataset_context=dataset_context,
            )
            for model_name, estimator in self._estimators.items():
                execution = self._executor.execute(
                    validation,
                    dataset_context,
                    model_name=model_name,
                    estimator=estimator,
                )
                pipeline_build_seconds += execution.build_time_seconds
                evaluation = self._ml_evaluator.evaluate(
                    execution.pipeline,
                    self._features,
                    self._target,
                    self._folds,
                    model_name=model_name,
                ).result
                model_outcomes.append(
                    ModelOutcome(
                        model_name=model_name,
                        status=evaluation.status,
                        primary_metric=evaluation.primary_metric_mean,
                        recall=self._mean(evaluation.metric_summary.get("recall")),
                        precision=self._mean(evaluation.metric_summary.get("precision")),
                        fit_seconds=evaluation.total_fit_seconds,
                        predict_seconds=evaluation.total_predict_seconds,
                    )
                )
        except CandidateEvaluationError:
            raise
        except (TypeError, ValueError) as exc:
            # Invalid data/plan combinations are useful feedback. Programming
            # errors outside these expected types remain visible to developers.
            raise CandidateEvaluationError(str(exc)) from exc

        return CandidateAssessment(
            plan_id=plan.plan_id,
            iteration=plan.iteration,
            validation_valid=True,
            execution_status="success",
            model_outcomes=tuple(model_outcomes),
            data_quality_score=quality.data_quality_score,
            preprocessing_seconds=pipeline_build_seconds + quality.evaluation_seconds,
        )

    @staticmethod
    def _mean(aggregate: MetricAggregate | None) -> float | None:
        return aggregate.mean if aggregate is not None else None
