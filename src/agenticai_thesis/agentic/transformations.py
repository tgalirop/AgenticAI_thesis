"""Extensible specifications and registry for allowlisted transformations.

Specifications validate declarative actions; they do not execute them. The future
Executor will resolve the same registry entries to deterministic implementation
objects, keeping LLM output strictly separated from arbitrary code execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from agenticai_thesis.agentic.domain import (
    ColumnKind,
    DatasetContext,
    TransformationAction,
    ValidationIssue,
)


class TransformationSpecification(ABC):
    """Abstract validation contract implemented by every safe transformation."""

    name: str
    allowed_column_kinds: frozenset[ColumnKind]
    requires_columns: bool = True

    def validate(
        self, action: TransformationAction, context: DatasetContext
    ) -> tuple[ValidationIssue, ...]:
        """Validate generic column constraints and transformation parameters."""

        issues: list[ValidationIssue] = []
        if self.requires_columns and not action.columns:
            issues.append(self._issue(action, "columns_required", "At least one column is required."))
        if not self.requires_columns and action.columns:
            issues.append(self._issue(action, "columns_not_allowed", "This action does not accept columns."))

        for column in action.columns:
            kind = context.column_types.get(column)
            if kind is not None and kind not in self.allowed_column_kinds:
                issues.append(
                    self._issue(
                        action,
                        "incompatible_column_type",
                        f"Column '{column}' has type '{kind}' which is not supported by '{self.name}'.",
                    )
                )
        issues.extend(self.validate_parameters(action))
        return tuple(issues)

    @abstractmethod
    def validate_parameters(self, action: TransformationAction) -> Iterable[ValidationIssue]:
        """Validate transformation-specific parameters."""

    @staticmethod
    def _issue(action: TransformationAction, code: str, message: str) -> ValidationIssue:
        return ValidationIssue(code=code, message=message, action_id=action.action_id)

    def _reject_unknown_parameters(
        self, action: TransformationAction, allowed: set[str]
    ) -> list[ValidationIssue]:
        unknown = sorted(set(action.parameters).difference(allowed))
        if not unknown:
            return []
        return [
            self._issue(
                action,
                "unknown_parameter",
                f"Unknown parameters for '{self.name}': {', '.join(unknown)}.",
            )
        ]


class NumericImputationSpecification(TransformationSpecification):
    """Allow median or mean imputation on numeric features."""

    name = "impute_numeric"
    allowed_column_kinds = frozenset({ColumnKind.NUMERIC})

    def validate_parameters(self, action: TransformationAction) -> Iterable[ValidationIssue]:
        issues = self._reject_unknown_parameters(action, {"strategy"})
        if action.parameters.get("strategy") not in {"median", "mean"}:
            issues.append(
                self._issue(action, "invalid_parameter", "strategy must be 'median' or 'mean'.")
            )
        return issues


class CategoricalImputationSpecification(TransformationSpecification):
    """Allow conservative imputation on categorical features."""

    name = "impute_categorical"
    allowed_column_kinds = frozenset({ColumnKind.CATEGORICAL})

    def validate_parameters(self, action: TransformationAction) -> Iterable[ValidationIssue]:
        issues = self._reject_unknown_parameters(action, {"strategy", "fill_value"})
        strategy = action.parameters.get("strategy")
        if strategy not in {"most_frequent", "constant"}:
            issues.append(
                self._issue(
                    action,
                    "invalid_parameter",
                    "strategy must be 'most_frequent' or 'constant'.",
                )
            )
        if strategy == "constant" and "fill_value" not in action.parameters:
            issues.append(
                self._issue(action, "missing_parameter", "constant strategy requires fill_value.")
            )
        return issues


class ScalingSpecification(TransformationSpecification):
    """Allow standard or robust scaling on numeric features."""

    name = "scale_numeric"
    allowed_column_kinds = frozenset({ColumnKind.NUMERIC})

    def validate_parameters(self, action: TransformationAction) -> Iterable[ValidationIssue]:
        issues = self._reject_unknown_parameters(action, {"method"})
        if action.parameters.get("method") not in {"standard", "robust"}:
            issues.append(
                self._issue(action, "invalid_parameter", "method must be 'standard' or 'robust'.")
            )
        return issues


class LogTransformSpecification(TransformationSpecification):
    """Allow log1p transformation of non-negative numeric features."""

    name = "log_transform"
    allowed_column_kinds = frozenset({ColumnKind.NUMERIC})

    def validate_parameters(self, action: TransformationAction) -> Iterable[ValidationIssue]:
        return self._reject_unknown_parameters(action, {"output_suffix"})


class OneHotEncodingSpecification(TransformationSpecification):
    """Allow safe one-hot encoding with unknown-category handling."""

    name = "one_hot_encode"
    allowed_column_kinds = frozenset({ColumnKind.CATEGORICAL})

    def validate_parameters(self, action: TransformationAction) -> Iterable[ValidationIssue]:
        issues = self._reject_unknown_parameters(action, {"handle_unknown"})
        if action.parameters.get("handle_unknown", "ignore") != "ignore":
            issues.append(
                self._issue(action, "unsafe_parameter", "handle_unknown must be 'ignore'.")
            )
        return issues


class ClassWeightSpecification(TransformationSpecification):
    """Allow model-level class weighting without selecting data columns."""

    name = "class_weight"
    allowed_column_kinds = frozenset()
    requires_columns = False

    def validate_parameters(self, action: TransformationAction) -> Iterable[ValidationIssue]:
        issues = self._reject_unknown_parameters(action, {"mode"})
        if action.parameters.get("mode") != "balanced":
            issues.append(self._issue(action, "invalid_parameter", "mode must be 'balanced'."))
        return issues


class SamplingSpecification(TransformationSpecification):
    """Allow fold-local sampling strategies; placement is checked by context."""

    name = "resample_classes"
    allowed_column_kinds = frozenset()
    requires_columns = False

    def validate_parameters(self, action: TransformationAction) -> Iterable[ValidationIssue]:
        issues = self._reject_unknown_parameters(action, {"method", "random_seed"})
        if action.parameters.get("method") not in {
            "random_undersampling",
            "random_oversampling",
            "smote",
        }:
            issues.append(
                self._issue(
                    action,
                    "invalid_parameter",
                    "method must be random_undersampling, random_oversampling, or smote.",
                )
            )
        seed = action.parameters.get("random_seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            issues.append(self._issue(action, "invalid_parameter", "random_seed must be an integer."))
        return issues


class TransformationRegistry:
    """Registry that supports extension without changing validator control flow."""

    def __init__(self, specifications: Iterable[TransformationSpecification] = ()) -> None:
        self._specifications: dict[str, TransformationSpecification] = {}
        for specification in specifications:
            self.register(specification)

    def register(self, specification: TransformationSpecification) -> None:
        """Register one unique specification by its stable action name."""

        if not specification.name:
            raise ValueError("Transformation specification name cannot be empty")
        if specification.name in self._specifications:
            raise ValueError(f"Transformation already registered: {specification.name}")
        self._specifications[specification.name] = specification

    def get(self, name: str) -> TransformationSpecification | None:
        """Resolve a specification without exposing the mutable backing mapping."""

        return self._specifications.get(name)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._specifications)

    @classmethod
    def default(cls) -> "TransformationRegistry":
        """Create the initial thesis whitelist of safe transformation types."""

        return cls(
            [
                NumericImputationSpecification(),
                CategoricalImputationSpecification(),
                ScalingSpecification(),
                LogTransformSpecification(),
                OneHotEncodingSpecification(),
                ClassWeightSpecification(),
                SamplingSpecification(),
            ]
        )
