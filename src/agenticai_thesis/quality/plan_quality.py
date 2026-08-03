"""Transformation-aware quality evaluation for Agent preprocessing plans."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from agenticai_thesis.agentic.candidate_evaluation import PlanQualityEvaluation
from agenticai_thesis.agentic.domain import DatasetContext, DatasetRole, ValidationResult
from agenticai_thesis.quality.quality_metrics import DataQualityEvaluator, DataQualityResult


class DataFrameQualityReportBuilder:
    """Build the minimal transparent report required by DataQualityEvaluator.

    This controlled experiment isolates completeness. Domain validity and
    cross-column consistency remain measured by the full development profiler;
    they are not fabricated for a small sampled modeling table.
    """

    def build(self, frame: pd.DataFrame) -> dict[str, object]:
        if frame.empty:
            raise ValueError("Cannot evaluate quality of an empty dataframe")
        return {
            "dimensions": {"rows": len(frame), "columns": len(frame.columns)},
            "columns": {
                column: {"null_count": int(frame[column].isna().sum())}
                for column in frame.columns
            },
            "invalid_values": {"checks": {}, "total_failures": 0},
            "consistency": {"checks": {}, "total_failures": 0},
            "duplicates": {"duplicate_rows": int(frame.duplicated().sum())},
        }


class TransformationAwarePlanQualityEvaluator:
    """Apply quality-relevant actions to a copy and calculate post-plan quality."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        report_builder: DataFrameQualityReportBuilder | None = None,
        quality_evaluator: DataQualityEvaluator | None = None,
    ) -> None:
        if frame.empty:
            raise ValueError("Quality evaluation frame cannot be empty")
        self._frame = frame.copy(deep=True)
        self._report_builder = report_builder or DataFrameQualityReportBuilder()
        self._quality_evaluator = quality_evaluator or DataQualityEvaluator()

    def initial_quality(self) -> DataQualityResult:
        """Return quality before any Agent-proposed transformation."""

        return self._quality_evaluator.evaluate(self._report_builder.build(self._frame))

    def initial_report(self) -> dict[str, object]:
        """Expose compact per-column missingness evidence for the Strategy Generator."""

        return self._report_builder.build(self._frame)

    def evaluate(
        self,
        *,
        validation: ValidationResult,
        dataset_context: DatasetContext,
    ) -> PlanQualityEvaluation:
        """Evaluate deterministic imputations; modeling-only actions are neutral."""

        if not validation.is_valid or validation.plan is None:
            raise ValueError("Plan quality evaluation requires a valid plan")
        if dataset_context.role != DatasetRole.DEVELOPMENT:
            raise ValueError("Plan quality evaluation is allowed only on development data")
        started = time.perf_counter()
        transformed = self._frame.copy(deep=True)

        for action in validation.plan.actions:
            if action.transformation == "impute_numeric":
                for column in action.columns:
                    values = pd.to_numeric(transformed[column], errors="coerce")
                    fill = values.median() if action.parameters["strategy"] == "median" else values.mean()
                    transformed[column] = values.fillna(fill)
            elif action.transformation == "impute_categorical":
                for column in action.columns:
                    if action.parameters["strategy"] == "constant":
                        fill = action.parameters["fill_value"]
                    else:
                        modes = transformed[column].mode(dropna=True)
                        if modes.empty:
                            raise ValueError(f"Cannot impute all-missing categorical column '{column}'")
                        fill = modes.iloc[0]
                    transformed[column] = transformed[column].fillna(fill)

        result = self._quality_evaluator.evaluate(self._report_builder.build(transformed))
        return PlanQualityEvaluation(
            data_quality_score=result.data_quality_score,
            evaluation_seconds=time.perf_counter() - started,
        )
