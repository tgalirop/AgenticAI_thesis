"""Object-oriented validation boundary for structured transformation plans."""

from __future__ import annotations

from collections.abc import Iterable

from agenticai_thesis.agentic.domain import (
    DatasetContext,
    DatasetRole,
    TransformationPlan,
    ValidationIssue,
    ValidationResult,
)
from agenticai_thesis.agentic.transformations import TransformationRegistry


class TransformationPlanValidator:
    """Validate a plan against schema, leakage, registry and parameter rules.

    The registry is injected rather than constructed internally. Tests and future
    deployments can therefore provide different allowlists without modifying this
    class or the LangGraph node that invokes it.
    """

    def __init__(
        self,
        registry: TransformationRegistry,
        *,
        allowed_transformations: Iterable[str] | None = None,
    ) -> None:
        self._registry = registry
        configured = set(allowed_transformations or registry.names)
        unknown_configuration = configured.difference(registry.names)
        if unknown_configuration:
            names = ", ".join(sorted(unknown_configuration))
            raise ValueError(f"Configured transformations are not registered: {names}")
        self._allowed_transformations = frozenset(configured)

    @property
    def allowed_transformations(self) -> frozenset[str]:
        """Expose the effective immutable whitelist for experiment logging."""

        return self._allowed_transformations

    def validate(
        self, plan: TransformationPlan, context: DatasetContext
    ) -> ValidationResult:
        """Return every detected issue so one retry can repair the complete plan."""

        issues: list[ValidationIssue] = []
        if context.role == DatasetRole.TEMPORAL_TEST:
            issues.append(
                ValidationIssue(
                    code="temporal_test_access_forbidden",
                    message="Transformation planning or execution on the temporal test set is forbidden.",
                )
            )
        if plan.dataset_id != context.dataset_id:
            issues.append(
                ValidationIssue(
                    code="dataset_mismatch",
                    message=(
                        f"Plan dataset_id '{plan.dataset_id}' does not match context "
                        f"dataset_id '{context.dataset_id}'."
                    ),
                )
            )

        protected = set(context.protected_columns) | {context.target_column}
        seen_action_signatures: set[tuple[object, ...]] = set()
        for action in plan.actions:
            if action.transformation not in self._allowed_transformations:
                issues.append(
                    ValidationIssue(
                        code="transformation_not_allowed",
                        message=f"Transformation '{action.transformation}' is not allowlisted.",
                        action_id=action.action_id,
                    )
                )
                continue

            missing_columns = sorted(set(action.columns).difference(context.column_types))
            for column in missing_columns:
                issues.append(
                    ValidationIssue(
                        code="column_not_found",
                        message=f"Column '{column}' does not exist in the dataset context.",
                        action_id=action.action_id,
                    )
                )

            protected_references = sorted(set(action.columns).intersection(protected))
            for column in protected_references:
                issues.append(
                    ValidationIssue(
                        code="protected_column_reference",
                        message=f"Protected column '{column}' cannot be transformed or used as a predictor.",
                        action_id=action.action_id,
                    )
                )

            # Sampling must be represented as a fold-local pipeline operation. It
            # can be selected from development context, but never run on validation
            # or temporal-test observations.
            if action.transformation == "resample_classes" and context.role == DatasetRole.VALIDATION_FOLD:
                issues.append(
                    ValidationIssue(
                        code="sampling_outside_training_forbidden",
                        message="Class resampling may be fitted only inside a training fold.",
                        action_id=action.action_id,
                    )
                )

            specification = self._registry.get(action.transformation)
            if specification is not None:
                issues.extend(specification.validate(action, context))

            # Repeated identical actions are almost always an LLM duplication and
            # can compound transformations unexpectedly, so reject them explicitly.
            signature = (
                action.transformation,
                action.columns,
                action.model_scope,
                repr(sorted(action.parameters.items())),
            )
            if signature in seen_action_signatures:
                issues.append(
                    ValidationIssue(
                        code="duplicate_action",
                        message="An identical transformation action already exists in this plan.",
                        action_id=action.action_id,
                    )
                )
            seen_action_signatures.add(signature)

        if issues:
            return ValidationResult(is_valid=False, issues=tuple(issues))
        return ValidationResult(is_valid=True, plan=plan)
