"""Strict domain models exchanged by the Agentic AI components.

These Pydantic models form the safety boundary between an LLM response and the
deterministic execution layer. Extra fields are forbidden and instances are
immutable, so a validated plan cannot be silently modified before execution.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ColumnKind(StrEnum):
    """Small semantic type system used by transformation specifications."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    TARGET = "target"


class DatasetRole(StrEnum):
    """Dataset partitions that may be presented to an Agent component."""

    DEVELOPMENT = "development"
    TRAINING_FOLD = "training_fold"
    VALIDATION_FOLD = "validation_fold"
    TEMPORAL_TEST = "temporal_test"


class DatasetContext(BaseModel):
    """Schema and partition metadata available during plan validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    role: DatasetRole
    column_types: dict[str, ColumnKind] = Field(min_length=1)
    target_column: str = Field(min_length=1)
    protected_columns: frozenset[str] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def target_must_exist_and_be_protected(self) -> "DatasetContext":
        """Ensure the supervised target is represented consistently in context."""

        if self.target_column not in self.column_types:
            raise ValueError("target_column must exist in column_types")
        if self.column_types[self.target_column] != ColumnKind.TARGET:
            raise ValueError("target_column must have ColumnKind.TARGET")
        return self


class TransformationAction(BaseModel):
    """One declarative transformation requested by the Strategy Generator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    transformation: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    columns: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)
    model_scope: Literal[
        "all",
        "logistic_regression",
        "decision_tree",
        "random_forest",
    ] = "all"

    @field_validator("columns")
    @classmethod
    def columns_must_be_unique(cls, columns: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous plans that repeat the same column in one action."""

        if len(columns) != len(set(columns)):
            raise ValueError("columns must not contain duplicates")
        if any(not column.strip() for column in columns):
            raise ValueError("column names cannot be empty")
        return columns


class TransformationPlan(BaseModel):
    """Complete structured strategy produced for one Agent iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    dataset_id: str = Field(min_length=1)
    iteration: int = Field(ge=1)
    objective: str = Field(min_length=1)
    actions: tuple[TransformationAction, ...] = ()

    @model_validator(mode="after")
    def action_ids_must_be_unique(self) -> "TransformationPlan":
        """Make validation issues and execution logs unambiguously addressable."""

        action_ids = [action.action_id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("action_id values must be unique within a plan")
        return self


class ValidationIssue(BaseModel):
    """One machine-readable reason why a plan cannot be executed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    action_id: str | None = None


class ValidationResult(BaseModel):
    """Immutable output returned by the plan validator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    is_valid: bool
    issues: tuple[ValidationIssue, ...] = ()
    plan: TransformationPlan | None = None

    @model_validator(mode="after")
    def result_state_must_be_coherent(self) -> "ValidationResult":
        """A valid result carries the plan; an invalid result carries issues."""

        if self.is_valid and (self.issues or self.plan is None):
            raise ValueError("A valid result requires a plan and no issues")
        if not self.is_valid and (not self.issues or self.plan is not None):
            raise ValueError("An invalid result requires issues and no plan")
        return self

