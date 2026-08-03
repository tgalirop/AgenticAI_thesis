"""Tests for deterministic and guarded transformation-plan execution."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from agenticai_thesis.agentic.domain import (
    ColumnKind,
    DatasetContext,
    DatasetRole,
    TransformationAction,
    TransformationPlan,
    ValidationIssue,
    ValidationResult,
)
from agenticai_thesis.agentic.execution import (
    ModelPipelineBuilder,
    SafeLog1pTransformer,
    TransformationFactoryRegistry,
)
from agenticai_thesis.agentic.executor import TransformationExecutor
from agenticai_thesis.agentic.transformations import TransformationRegistry
from agenticai_thesis.agentic.validator import TransformationPlanValidator


def _context(role: DatasetRole = DatasetRole.DEVELOPMENT) -> DatasetContext:
    return DatasetContext(
        dataset_id="development-v1",
        role=role,
        column_types={
            "amount": ColumnKind.NUMERIC,
            "type": ColumnKind.CATEGORICAL,
            "isFraud": ColumnKind.TARGET,
        },
        target_column="isFraud",
        protected_columns=frozenset({"isFraud"}),
    )


def _executable_plan() -> TransformationPlan:
    """Build a representative plan covering preprocessing and resampling."""

    return TransformationPlan(
        plan_id="plan_001",
        dataset_id="development-v1",
        iteration=1,
        objective="Prepare Logistic Regression features.",
        actions=(
            TransformationAction(
                action_id="impute_amount",
                transformation="impute_numeric",
                columns=("amount",),
                parameters={"strategy": "median"},
                rationale="Replace missing transaction amounts.",
            ),
            TransformationAction(
                action_id="scale_amount",
                transformation="scale_numeric",
                columns=("amount",),
                parameters={"method": "robust"},
                rationale="Reduce sensitivity to large amounts.",
                model_scope="logistic_regression",
            ),
            TransformationAction(
                action_id="impute_type",
                transformation="impute_categorical",
                columns=("type",),
                parameters={"strategy": "most_frequent"},
                rationale="Replace missing transaction types.",
            ),
            TransformationAction(
                action_id="encode_type",
                transformation="one_hot_encode",
                columns=("type",),
                parameters={"handle_unknown": "ignore"},
                rationale="Create non-ordinal type indicators.",
            ),
            TransformationAction(
                action_id="weight_classes",
                transformation="class_weight",
                parameters={"mode": "balanced"},
                rationale="Account for fraud class imbalance.",
            ),
            TransformationAction(
                action_id="oversample_classes",
                transformation="resample_classes",
                parameters={"method": "random_oversampling", "random_seed": 42},
                rationale="Resample only inside each training fit.",
            ),
        ),
    )


def _validated(plan: TransformationPlan, context: DatasetContext) -> ValidationResult:
    return TransformationPlanValidator(TransformationRegistry.default()).validate(plan, context)


def _executor() -> TransformationExecutor:
    registry = TransformationFactoryRegistry.default()
    return TransformationExecutor(ModelPipelineBuilder(registry))


def _training_data() -> tuple[pd.DataFrame, np.ndarray]:
    x = pd.DataFrame(
        {
            "amount": [10.0, 20.0, None, 40.0, 100.0, 120.0, 140.0, 160.0],
            "type": ["PAYMENT", "PAYMENT", "CASH_OUT", None, "TRANSFER", "TRANSFER", "CASH_OUT", "TRANSFER"],
        }
    )
    y = np.asarray([0, 0, 0, 0, 0, 0, 1, 1])
    return x, y


def test_executor_builds_pipeline_that_fits_and_predicts() -> None:
    context = _context()
    validation = _validated(_executable_plan(), context)
    result = _executor().execute(
        validation,
        context,
        model_name="logistic_regression",
        estimator=LogisticRegression(max_iter=200, random_state=42),
    )
    x, y = _training_data()
    result.pipeline.fit(x, y)
    probabilities = result.pipeline.predict_proba(x)[:, 1]

    assert result.status == "built"
    assert len(result.applied_actions) == 6
    assert probabilities.shape == (len(x),)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert result.pipeline.named_steps["classifier"].class_weight == "balanced"
    assert "sampler" in result.pipeline.named_steps


def test_same_validated_plan_and_seed_produce_identical_predictions() -> None:
    context = _context()
    validation = _validated(_executable_plan(), context)
    x, y = _training_data()
    predictions = []
    for _ in range(2):
        result = _executor().execute(
            validation,
            context,
            model_name="logistic_regression",
            estimator=LogisticRegression(max_iter=200, random_state=42),
        )
        result.pipeline.fit(x, y)
        predictions.append(result.pipeline.predict_proba(x)[:, 1])
    np.testing.assert_allclose(predictions[0], predictions[1])


def test_model_scoped_action_is_skipped_for_another_model() -> None:
    plan = TransformationPlan(
        plan_id="plan_scope",
        dataset_id="development-v1",
        iteration=1,
        objective="Test model scoping.",
        actions=(
            TransformationAction(
                action_id="scale_lr",
                transformation="scale_numeric",
                columns=("amount",),
                parameters={"method": "standard"},
                rationale="Only Logistic Regression needs scaling.",
                model_scope="logistic_regression",
            ),
        ),
    )
    context = _context()
    result = _executor().execute(
        _validated(plan, context),
        context,
        model_name="decision_tree",
        estimator=LogisticRegression(max_iter=100),
    )
    assert result.applied_actions == ()
    assert "preprocessing" not in result.pipeline.named_steps


def test_executor_rejects_invalid_validation_result() -> None:
    invalid = ValidationResult(
        is_valid=False,
        issues=(ValidationIssue(code="invalid", message="Invalid test plan."),),
    )
    with pytest.raises(ValueError, match="valid ValidationResult"):
        _executor().execute(
            invalid,
            _context(),
            model_name="logistic_regression",
            estimator=LogisticRegression(),
        )


@pytest.mark.parametrize("role", [DatasetRole.VALIDATION_FOLD, DatasetRole.TEMPORAL_TEST])
def test_executor_rejects_forbidden_dataset_roles(role: DatasetRole) -> None:
    # Build a coherent validated result directly so this test exercises the
    # Executor's independent defence-in-depth check rather than the Validator.
    plan = _executable_plan()
    validation = ValidationResult(is_valid=True, plan=plan)
    with pytest.raises(ValueError, match="forbidden"):
        _executor().execute(
            validation,
            _context(role),
            model_name="logistic_regression",
            estimator=LogisticRegression(),
        )


def test_safe_log_transform_rejects_negative_input() -> None:
    transformer = SafeLog1pTransformer()
    with pytest.raises(ValueError, match="non-negative"):
        transformer.fit(np.asarray([[1.0], [-0.1]]))


def test_builder_fails_closed_when_factory_is_missing() -> None:
    context = _context()
    validation = _validated(_executable_plan(), context)
    empty_executor = TransformationExecutor(
        ModelPipelineBuilder(TransformationFactoryRegistry())
    )
    with pytest.raises(ValueError, match="No executable factory"):
        empty_executor.execute(
            validation,
            context,
            model_name="logistic_regression",
            estimator=LogisticRegression(),
        )
