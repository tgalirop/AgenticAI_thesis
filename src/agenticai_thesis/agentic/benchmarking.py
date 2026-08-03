"""Artifact-producing benchmarks for validated Agentic plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from agenticai_thesis.agentic.domain import DatasetContext, ValidationResult
from agenticai_thesis.agentic.evaluators import MachineLearningEvaluator
from agenticai_thesis.agentic.executor import TransformationExecutor
from agenticai_thesis.modeling.benchmark import BenchmarkResult
from agenticai_thesis.modeling.cross_validation import CrossValidationFoldSet
from agenticai_thesis.modeling.metrics import compute_classification_metrics
from agenticai_thesis.modeling.models import model_requires_scaling
from agenticai_thesis.modeling.preprocessing import MODEL_FEATURES, build_preprocessor


class AgenticPlanBenchmark:
    """Materialize fold metrics and OOF predictions for one locked plan."""

    def __init__(
        self,
        *,
        executor: TransformationExecutor,
        evaluator: MachineLearningEvaluator,
        estimators: Mapping[str, ClassifierMixin],
        features: pd.DataFrame,
        target: np.ndarray,
        folds: CrossValidationFoldSet,
        context: DatasetContext,
    ) -> None:
        self._executor = executor
        self._evaluator = evaluator
        self._estimators = dict(estimators)
        self._features = features
        self._target = np.asarray(target, dtype=int)
        self._folds = folds
        self._context = context

    def run(self, validation: ValidationResult) -> BenchmarkResult:
        metric_rows: list[dict[str, Any]] = []
        predictions: list[pd.DataFrame] = []
        for model_name, estimator in self._estimators.items():
            execution = self._executor.execute(
                validation,
                self._context,
                model_name=model_name,
                estimator=estimator,
            )
            output = self._evaluator.evaluate(
                execution.pipeline,
                self._features,
                self._target,
                self._folds,
                model_name=model_name,
            )
            for fold in output.result.fold_results:
                if fold.status != "success" or fold.metrics is None:
                    continue
                metric_rows.append(
                    {
                        "model": model_name,
                        "repeat": fold.repeat,
                        "fold": fold.fold,
                        "train_rows": fold.train_rows,
                        "validation_rows": fold.validation_rows,
                        "fit_seconds": fold.fit_seconds,
                        "predict_seconds": fold.predict_seconds,
                        **fold.metrics.model_dump(),
                    }
                )
            predictions.append(output.predictions)
        return BenchmarkResult(
            fold_metrics=pd.DataFrame.from_records(metric_rows),
            predictions=pd.concat(predictions, ignore_index=True),
        )


@dataclass(frozen=True)
class TemporalEvaluationOutput:
    """Final untouched-holdout metrics and prediction records."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame


class TemporalHoldoutComparator:
    """Compare fixed conventional and selected Agentic pipelines exactly once."""

    def __init__(
        self,
        *,
        executor: TransformationExecutor,
        estimators: Mapping[str, ClassifierMixin],
        context: DatasetContext,
        threshold: float,
    ) -> None:
        self._executor = executor
        self._estimators = dict(estimators)
        self._context = context
        self._threshold = threshold

    def evaluate(
        self,
        *,
        validation: ValidationResult,
        development: pd.DataFrame,
        temporal_test: pd.DataFrame,
        target_column: str,
    ) -> TemporalEvaluationOutput:
        x_train = development[MODEL_FEATURES]
        y_train = development[target_column].astype(int).to_numpy()
        x_test = temporal_test[MODEL_FEATURES]
        y_test = temporal_test[target_column].astype(int).to_numpy()
        metric_rows: list[dict[str, Any]] = []
        prediction_rows: list[pd.DataFrame] = []

        for model_name, estimator in self._estimators.items():
            conventional = Pipeline(
                [
                    ("preprocessing", build_preprocessor(scale_numeric=model_requires_scaling(model_name))),
                    ("classifier", estimator),
                ]
            )
            agentic = self._executor.execute(
                validation,
                self._context,
                model_name=model_name,
                estimator=estimator,
            ).pipeline
            for pipeline_name, pipeline in (("conventional", conventional), ("agentic", agentic)):
                pipeline.fit(x_train, y_train)
                scores = pipeline.predict_proba(x_test)[:, 1]
                metric_rows.append(
                    {
                        "pipeline": pipeline_name,
                        "model": model_name,
                        "train_rows": len(x_train),
                        "temporal_test_rows": len(x_test),
                        **compute_classification_metrics(y_test, scores, threshold=self._threshold),
                    }
                )
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "pipeline": pipeline_name,
                            "model": model_name,
                            "row_index": np.arange(len(x_test)),
                            "y_true": y_test,
                            "y_score": scores,
                        }
                    )
                )
        return TemporalEvaluationOutput(
            metrics=pd.DataFrame.from_records(metric_rows),
            predictions=pd.concat(prediction_rows, ignore_index=True),
        )
