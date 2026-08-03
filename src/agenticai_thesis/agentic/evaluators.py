"""Object-oriented evaluation of Agent-generated machine-learning pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import clone

from agenticai_thesis.modeling.cross_validation import CrossValidationFoldSet
from agenticai_thesis.modeling.metrics import compute_classification_metrics


class EvaluationConfig(BaseModel):
    """Immutable settings shared by all model evaluations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_metric: Literal["pr_auc", "roc_auc", "f1", "recall", "precision"] = "pr_auc"
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ClassificationMetrics(BaseModel):
    """Typed classification metrics for one held-out fold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accuracy: float
    recall: float
    specificity: float
    precision: float
    f1: float
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int


class FoldEvaluationResult(BaseModel):
    """Outcome and resource timings of one CV fold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repeat: int
    fold: int
    train_rows: int
    validation_rows: int
    fit_seconds: float
    predict_seconds: float
    status: Literal["success", "error"]
    metrics: ClassificationMetrics | None = None
    error_type: str | None = None
    error_message: str | None = None


class MetricAggregate(BaseModel):
    """Descriptive summary of one metric across successful folds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: float
    std: float
    minimum: float
    maximum: float


class ModelEvaluationResult(BaseModel):
    """Agent-state-safe summary of a complete model evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    status: Literal["success", "partial_failure", "error"]
    primary_metric: str
    primary_metric_mean: float | None
    fold_results: tuple[FoldEvaluationResult, ...]
    metric_summary: dict[str, MetricAggregate]
    successful_folds: int
    failed_folds: int
    total_fit_seconds: float
    total_predict_seconds: float


@dataclass(frozen=True)
class EvaluationOutput:
    """Typed result plus detailed OOF predictions kept outside Agent state."""

    result: ModelEvaluationResult
    predictions: pd.DataFrame


class MachineLearningEvaluator:
    """Evaluate one unfitted pipeline on an injected immutable shared fold set."""

    _SUMMARY_METRICS = (
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
    )

    def __init__(self, config: EvaluationConfig) -> None:
        self._config = config

    def evaluate(
        self,
        pipeline: Any,
        x: pd.DataFrame,
        y: np.ndarray,
        folds: CrossValidationFoldSet,
        *,
        model_name: str,
    ) -> EvaluationOutput:
        """Fit fresh pipeline clones and collect fold-level metrics and errors.

        A failure in one fold is recorded rather than destroying the complete Agent
        iteration. The feedback policy can then decide whether a partial result is
        acceptable or should trigger a revised transformation strategy.
        """

        target = np.asarray(y, dtype=int)
        if len(x) != len(target):
            raise ValueError("Feature and target row counts do not match")
        folds.validate_target(target)
        fold_results: list[FoldEvaluationResult] = []
        prediction_frames: list[pd.DataFrame] = []

        for split in folds.splits:
            fit_seconds = 0.0
            predict_seconds = 0.0
            try:
                fold_pipeline = clone(pipeline)
                fit_started = time.perf_counter()
                fold_pipeline.fit(x.iloc[split.train_indices], target[split.train_indices])
                fit_seconds = time.perf_counter() - fit_started

                predict_started = time.perf_counter()
                scores = fold_pipeline.predict_proba(x.iloc[split.validation_indices])[:, 1]
                predict_seconds = time.perf_counter() - predict_started
                raw_metrics = compute_classification_metrics(
                    target[split.validation_indices],
                    scores,
                    threshold=self._config.decision_threshold,
                )
                metrics = ClassificationMetrics.model_validate(raw_metrics)
                fold_results.append(
                    FoldEvaluationResult(
                        repeat=split.repeat,
                        fold=split.fold,
                        train_rows=len(split.train_indices),
                        validation_rows=len(split.validation_indices),
                        fit_seconds=fit_seconds,
                        predict_seconds=predict_seconds,
                        status="success",
                        metrics=metrics,
                    )
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "model": model_name,
                            "repeat": split.repeat,
                            "fold": split.fold,
                            "row_index": split.validation_indices,
                            "y_true": target[split.validation_indices],
                            "y_score": scores,
                        }
                    )
                )
            except Exception as error:  # The typed record is deliberate Agent feedback.
                fold_results.append(
                    FoldEvaluationResult(
                        repeat=split.repeat,
                        fold=split.fold,
                        train_rows=len(split.train_indices),
                        validation_rows=len(split.validation_indices),
                        fit_seconds=fit_seconds,
                        predict_seconds=predict_seconds,
                        status="error",
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                )

        successful = [fold for fold in fold_results if fold.status == "success"]
        failed_count = len(fold_results) - len(successful)
        summary = self._summarize(successful)
        if not successful:
            status = "error"
        elif failed_count:
            status = "partial_failure"
        else:
            status = "success"
        primary = summary.get(self._config.primary_metric)
        predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame(columns=["model", "repeat", "fold", "row_index", "y_true", "y_score"])
        )
        result = ModelEvaluationResult(
            model_name=model_name,
            status=status,
            primary_metric=self._config.primary_metric,
            primary_metric_mean=primary.mean if primary else None,
            fold_results=tuple(fold_results),
            metric_summary=summary,
            successful_folds=len(successful),
            failed_folds=failed_count,
            total_fit_seconds=sum(fold.fit_seconds for fold in fold_results),
            total_predict_seconds=sum(fold.predict_seconds for fold in fold_results),
        )
        return EvaluationOutput(result=result, predictions=predictions)

    def _summarize(
        self, successful_folds: list[FoldEvaluationResult]
    ) -> dict[str, MetricAggregate]:
        """Aggregate only successful folds while retaining failures separately."""

        summary: dict[str, MetricAggregate] = {}
        for metric_name in self._SUMMARY_METRICS:
            values = np.asarray(
                [getattr(fold.metrics, metric_name) for fold in successful_folds if fold.metrics],
                dtype=float,
            )
            if not len(values):
                continue
            summary[metric_name] = MetricAggregate(
                mean=float(values.mean()),
                std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                minimum=float(values.min()),
                maximum=float(values.max()),
            )
        return summary
