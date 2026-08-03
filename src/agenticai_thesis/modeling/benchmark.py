"""Benchmark all conventional models on identical cross-validation folds."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

from agenticai_thesis.modeling.cross_validation import CrossValidationFoldProvider
from agenticai_thesis.modeling.metrics import compute_classification_metrics
from agenticai_thesis.modeling.models import create_estimator, model_requires_scaling
from agenticai_thesis.modeling.preprocessing import (
    MODEL_FEATURES,
    build_preprocessor,
    validate_model_features,
)


@dataclass(frozen=True)
class BenchmarkResult:
    """Fold-level metrics and repeated out-of-fold prediction records."""

    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame


def load_reproducible_sample(
    development_path: str | Path,
    *,
    target_column: str,
    negative_to_positive_ratio: int,
    random_seed: int,
) -> pd.DataFrame:
    """Load all fraud rows and a fixed random sample of normal transactions.

    Sampling occurs once, before constructing CV folds, and the resulting table is
    passed unchanged to every estimator.  The temporal test file is neither an
    argument nor read by this function, which protects the final holdout.
    """

    if negative_to_positive_ratio < 1:
        raise ValueError("negative_to_positive_ratio must be at least 1")

    required_columns = [*MODEL_FEATURES, target_column]
    lazy = pl.scan_parquet(development_path).select(required_columns)
    validate_model_features(lazy.collect_schema().names())

    fraud = lazy.filter(pl.col(target_column) == 1).collect(engine="streaming")
    normal = lazy.filter(pl.col(target_column) == 0).collect(engine="streaming")
    requested_normal = fraud.height * negative_to_positive_ratio
    if fraud.height == 0:
        raise ValueError("Development dataset contains no fraud observations")
    if requested_normal > normal.height:
        raise ValueError("Requested more normal observations than are available")

    sampled_normal = normal.sample(n=requested_normal, seed=random_seed, shuffle=True)
    # The final shuffle prevents class-block ordering while remaining deterministic.
    sample = pl.concat([fraud, sampled_normal]).sample(
        fraction=1.0, seed=random_seed, shuffle=True
    )
    return sample.to_pandas()


def benchmark_models(
    sample: pd.DataFrame,
    *,
    model_names: Sequence[str],
    model_parameters: Mapping[str, Mapping[str, Any]],
    target_column: str,
    folds: int,
    repeats: int,
    random_seed: int,
    threshold: float,
) -> BenchmarkResult:
    """Evaluate every model on the exact same materialised repeated CV splits."""

    if folds < 2 or repeats < 1:
        raise ValueError("Cross-validation requires folds >= 2 and repeats >= 1")
    validate_model_features(sample.columns.tolist())
    if target_column not in sample:
        raise ValueError(f"Target column not found: {target_column}")

    x = sample[MODEL_FEATURES]
    y = sample[target_column].astype(int).to_numpy()
    # Materialising indices once is the key fairness guarantee: every estimator
    # sees exactly the same training and validation observations in every run.
    shared_folds = CrossValidationFoldProvider(
        folds=folds,
        repeats=repeats,
        random_seed=random_seed,
    ).create(y)
    metric_records: list[dict[str, Any]] = []
    prediction_records: list[pd.DataFrame] = []

    for model_name in model_names:
        for split in shared_folds.splits:
            train_indices = split.train_indices
            validation_indices = split.validation_indices
            repeat_index = split.repeat
            fold_index = split.fold
            estimator = create_estimator(
                model_name,
                random_seed=random_seed,
                parameters=model_parameters.get(model_name, {}),
            )
            pipeline = Pipeline(
                [
                    ("preprocessing", build_preprocessor(scale_numeric=model_requires_scaling(model_name))),
                    ("classifier", estimator),
                ]
            )

            fit_started = time.perf_counter()
            pipeline.fit(x.iloc[train_indices], y[train_indices])
            fit_seconds = time.perf_counter() - fit_started
            predict_started = time.perf_counter()
            scores = pipeline.predict_proba(x.iloc[validation_indices])[:, 1]
            predict_seconds = time.perf_counter() - predict_started

            metrics = compute_classification_metrics(
                y[validation_indices], scores, threshold=threshold
            )
            metric_records.append(
                {
                    "model": model_name,
                    "repeat": repeat_index,
                    "fold": fold_index,
                    "train_rows": len(train_indices),
                    "validation_rows": len(validation_indices),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    **metrics,
                }
            )
            prediction_records.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "row_index": validation_indices,
                        "y_true": y[validation_indices],
                        "y_score": scores,
                    }
                )
            )

    return BenchmarkResult(
        fold_metrics=pd.DataFrame.from_records(metric_records),
        predictions=pd.concat(prediction_records, ignore_index=True),
    )


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise fold-level performance without discarding the raw results."""

    metric_columns = [
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "fit_seconds",
        "predict_seconds",
    ]
    summary = fold_metrics.groupby("model", sort=False)[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index()
