"""Tests for structured transformation plans and their safety validator."""

import pytest
from pydantic import ValidationError

from agenticai_thesis.agentic.domain import (
    ColumnKind,
    DatasetContext,
    DatasetRole,
    TransformationAction,
    TransformationPlan,
)
from agenticai_thesis.agentic.transformations import TransformationRegistry
from agenticai_thesis.agentic.validator import TransformationPlanValidator


def _context(role: DatasetRole = DatasetRole.DEVELOPMENT) -> DatasetContext:
    """Create the minimal schema exposed to the Agent during validation."""

    return DatasetContext(
        dataset_id="paysim-development-v1",
        role=role,
        column_types={
            "amount": ColumnKind.NUMERIC,
            "log_amount": ColumnKind.NUMERIC,
            "type": ColumnKind.CATEGORICAL,
            "is_transfer": ColumnKind.BINARY,
            "isFraud": ColumnKind.TARGET,
        },
        target_column="isFraud",
        protected_columns=frozenset({"isFraud"}),
    )


def _plan(*actions: TransformationAction, dataset_id: str = "paysim-development-v1") -> TransformationPlan:
    return TransformationPlan(
        plan_id="plan_001",
        dataset_id=dataset_id,
        iteration=1,
        objective="Prepare model-specific fraud-detection features.",
        actions=actions,
    )


def _validator() -> TransformationPlanValidator:
    return TransformationPlanValidator(TransformationRegistry.default())


def test_valid_plan_is_returned_unchanged() -> None:
    plan = _plan(
        TransformationAction(
            action_id="scale_1",
            transformation="scale_numeric",
            columns=("amount", "log_amount"),
            parameters={"method": "robust"},
            rationale="Reduce sensitivity to extreme transaction amounts.",
            model_scope="logistic_regression",
        ),
        TransformationAction(
            action_id="encode_1",
            transformation="one_hot_encode",
            columns=("type",),
            parameters={"handle_unknown": "ignore"},
            rationale="Encode transaction type without imposing an ordinal relation.",
        ),
    )

    result = _validator().validate(plan, _context())
    assert result.is_valid is True
    assert result.plan is plan
    assert result.issues == ()


def test_unknown_transformation_is_rejected() -> None:
    plan = _plan(
        TransformationAction(
            action_id="unsafe_1",
            transformation="execute_python",
            columns=("amount",),
            rationale="Attempt arbitrary code execution.",
        )
    )
    result = _validator().validate(plan, _context())
    assert result.is_valid is False
    assert {issue.code for issue in result.issues} == {"transformation_not_allowed"}


def test_missing_and_protected_columns_are_reported_together() -> None:
    plan = _plan(
        TransformationAction(
            action_id="scale_1",
            transformation="scale_numeric",
            columns=("missing_column", "isFraud"),
            parameters={"method": "standard"},
            rationale="Invalid schema references for testing.",
        )
    )
    result = _validator().validate(plan, _context())
    codes = {issue.code for issue in result.issues}
    assert "column_not_found" in codes
    assert "protected_column_reference" in codes
    assert "incompatible_column_type" in codes


def test_incompatible_type_and_invalid_parameters_are_rejected() -> None:
    plan = _plan(
        TransformationAction(
            action_id="impute_1",
            transformation="impute_numeric",
            columns=("type",),
            parameters={"strategy": "magic", "unexpected": True},
            rationale="Deliberately invalid action.",
        )
    )
    result = _validator().validate(plan, _context())
    codes = {issue.code for issue in result.issues}
    assert codes == {"incompatible_column_type", "invalid_parameter", "unknown_parameter"}


def test_temporal_test_context_is_never_validated_for_execution() -> None:
    result = _validator().validate(_plan(), _context(DatasetRole.TEMPORAL_TEST))
    assert result.is_valid is False
    assert result.issues[0].code == "temporal_test_access_forbidden"


def test_sampling_is_rejected_on_validation_fold() -> None:
    plan = _plan(
        TransformationAction(
            action_id="sample_1",
            transformation="resample_classes",
            parameters={"method": "smote", "random_seed": 42},
            rationale="Balance the training observations.",
        )
    )
    result = _validator().validate(plan, _context(DatasetRole.VALIDATION_FOLD))
    assert "sampling_outside_training_forbidden" in {issue.code for issue in result.issues}


def test_duplicate_actions_are_rejected() -> None:
    action_1 = TransformationAction(
        action_id="weight_1",
        transformation="class_weight",
        parameters={"mode": "balanced"},
        rationale="Handle class imbalance.",
    )
    action_2 = action_1.model_copy(update={"action_id": "weight_2"})
    result = _validator().validate(_plan(action_1, action_2), _context())
    assert "duplicate_action" in {issue.code for issue in result.issues}


def test_plan_schema_forbids_extra_llm_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TransformationAction.model_validate(
            {
                "action_id": "scale_1",
                "transformation": "scale_numeric",
                "columns": ["amount"],
                "parameters": {"method": "standard"},
                "rationale": "Scale amount.",
                "python_code": "import os",
            }
        )


def test_registry_rejects_duplicate_specification_names() -> None:
    registry = TransformationRegistry.default()
    duplicate = TransformationRegistry.default().get("scale_numeric")
    assert duplicate is not None
    with pytest.raises(ValueError, match="already registered"):
        registry.register(duplicate)
