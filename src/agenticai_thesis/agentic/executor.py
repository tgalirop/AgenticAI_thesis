"""Guarded object-oriented entry point for deterministic plan execution."""

from __future__ import annotations

import time
from dataclasses import dataclass

from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import ClassifierMixin

from agenticai_thesis.agentic.domain import DatasetContext, DatasetRole, ValidationResult
from agenticai_thesis.agentic.execution import AppliedAction, ModelPipelineBuilder


@dataclass(frozen=True)
class ExecutionResult:
    """Auditable result of converting one validated plan into an ML pipeline."""

    plan_id: str
    dataset_id: str
    model_name: str
    pipeline: ImbalancedPipeline
    applied_actions: tuple[AppliedAction, ...]
    build_time_seconds: float
    status: str = "built"


class TransformationExecutor:
    """Execute only coherent, validated plans outside forbidden dataset roles."""

    def __init__(self, pipeline_builder: ModelPipelineBuilder) -> None:
        self._pipeline_builder = pipeline_builder

    def execute(
        self,
        validation: ValidationResult,
        context: DatasetContext,
        *,
        model_name: str,
        estimator: ClassifierMixin,
    ) -> ExecutionResult:
        """Build a deterministic model pipeline or fail closed before execution."""

        if not validation.is_valid or validation.plan is None:
            raise ValueError("TransformationExecutor requires a valid ValidationResult")
        plan = validation.plan
        if plan.dataset_id != context.dataset_id:
            raise ValueError("Validated plan and execution context refer to different datasets")
        if context.role in {DatasetRole.VALIDATION_FOLD, DatasetRole.TEMPORAL_TEST}:
            raise ValueError(f"Pipeline construction is forbidden for dataset role '{context.role}'")

        started = time.perf_counter()
        pipeline, applied_actions = self._pipeline_builder.build(
            plan,
            model_name=model_name,
            estimator=estimator,
        )
        return ExecutionResult(
            plan_id=plan.plan_id,
            dataset_id=context.dataset_id,
            model_name=model_name,
            pipeline=pipeline,
            applied_actions=applied_actions,
            build_time_seconds=time.perf_counter() - started,
        )
