"""Tests for deterministic controlled degradation and plan quality scoring."""

import pandas as pd
import pytest

from agenticai_thesis.agentic.domain import (
    ColumnKind,
    DatasetContext,
    DatasetRole,
    TransformationAction,
    TransformationPlan,
)
from agenticai_thesis.agentic.transformations import TransformationRegistry
from agenticai_thesis.agentic.validator import TransformationPlanValidator
from agenticai_thesis.quality.degradation import (
    ControlledDataDegrader,
    ControlledDegradationConfig,
)
from agenticai_thesis.quality.plan_quality import TransformationAwarePlanQualityEvaluator


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [float(value) for value in range(1, 21)],
            "type": ["PAYMENT", "TRANSFER"] * 10,
            "isFraud": [0, 1] * 10,
        }
    )


def _context() -> DatasetContext:
    return DatasetContext(
        dataset_id="development-v1",
        role=DatasetRole.DEVELOPMENT,
        column_types={
            "amount": ColumnKind.NUMERIC,
            "type": ColumnKind.CATEGORICAL,
            "isFraud": ColumnKind.TARGET,
        },
        target_column="isFraud",
        protected_columns=frozenset({"isFraud"}),
    )


def test_degradation_is_reproducible_and_preserves_input() -> None:
    original = _frame()
    config = ControlledDegradationConfig({"amount": 0.20, "type": 0.10}, random_seed=42)
    degrader = ControlledDataDegrader(config, protected_columns=frozenset({"isFraud"}))
    first = degrader.degrade(original)
    second = degrader.degrade(original)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.affected_rows_by_column == {"amount": 4, "type": 2}
    assert original.isna().sum().sum() == 0


def test_degrader_rejects_protected_target() -> None:
    with pytest.raises(ValueError, match="Protected"):
        ControlledDataDegrader(
            ControlledDegradationConfig({"isFraud": 0.10}),
            protected_columns=frozenset({"isFraud"}),
        )


def test_imputation_plan_improves_measured_completeness() -> None:
    degraded = ControlledDataDegrader(
        ControlledDegradationConfig({"amount": 0.20, "type": 0.10}, random_seed=42),
        protected_columns=frozenset({"isFraud"}),
    ).degrade(_frame()).frame
    evaluator = TransformationAwarePlanQualityEvaluator(degraded)
    initial = evaluator.initial_quality()
    plan = TransformationPlan(
        plan_id="repair_001",
        dataset_id="development-v1",
        iteration=1,
        objective="Repair controlled missingness.",
        actions=(
            TransformationAction(
                action_id="numeric",
                transformation="impute_numeric",
                columns=("amount",),
                parameters={"strategy": "median"},
                rationale="Repair numeric missingness.",
            ),
            TransformationAction(
                action_id="categorical",
                transformation="impute_categorical",
                columns=("type",),
                parameters={"strategy": "most_frequent"},
                rationale="Repair categorical missingness.",
            ),
        ),
    )
    validation = TransformationPlanValidator(TransformationRegistry.default()).validate(
        plan, _context()
    )
    repaired = evaluator.evaluate(validation=validation, dataset_context=_context())

    assert initial.completeness < 1.0
    assert repaired.data_quality_score > initial.data_quality_score
