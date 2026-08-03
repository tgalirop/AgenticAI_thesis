"""Deterministic transformation factories and model-pipeline construction.

No class in this module accepts source code, imports, callables, or shell commands
from an Agent plan. A validated action name is resolved through an injected
registry to a concrete, pre-implemented factory.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from agenticai_thesis.agentic.domain import TransformationAction, TransformationPlan


class ExecutionArtifactKind(StrEnum):
    """Locations at which a built operation can enter an ML pipeline."""

    COLUMN_TRANSFORMER = "column_transformer"
    SAMPLER = "sampler"
    ESTIMATOR_PARAMETERS = "estimator_parameters"


@dataclass(frozen=True)
class ExecutionArtifact:
    """Typed result built by a deterministic transformation factory."""

    kind: ExecutionArtifactKind
    value: Any


@dataclass(frozen=True)
class AppliedAction:
    """Audit record describing one plan action included in a model pipeline."""

    action_id: str
    transformation: str
    columns: tuple[str, ...]
    model_scope: str


class SafeLog1pTransformer(TransformerMixin, BaseEstimator):
    """Apply log1p only to finite, non-negative numeric input.

    ``FunctionTransformer(np.log1p)`` would silently produce NaN values for inputs
    below -1. Explicit validation turns a bad data/strategy combination into an
    auditable execution error instead of allowing corrupted features downstream.
    """

    def fit(self, x: Any, y: Any = None) -> "SafeLog1pTransformer":
        values = check_array(x, dtype=float, ensure_all_finite=True)
        if np.any(values < 0):
            raise ValueError("log_transform requires non-negative values")
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, x: Any) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        values = check_array(x, dtype=float, ensure_all_finite=True)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Input feature count changed after fitting log transformer")
        if np.any(values < 0):
            raise ValueError("log_transform requires non-negative values")
        return np.log1p(values)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


class TransformationFactory(ABC):
    """Factory contract for one allowlisted executable transformation."""

    name: str

    @abstractmethod
    def build(self, action: TransformationAction) -> ExecutionArtifact:
        """Build a trusted artifact exclusively from validated parameters."""


class NumericImputationFactory(TransformationFactory):
    name = "impute_numeric"

    def build(self, action: TransformationAction) -> ExecutionArtifact:
        return ExecutionArtifact(
            ExecutionArtifactKind.COLUMN_TRANSFORMER,
            SimpleImputer(strategy=action.parameters["strategy"]),
        )


class CategoricalImputationFactory(TransformationFactory):
    name = "impute_categorical"

    def build(self, action: TransformationAction) -> ExecutionArtifact:
        strategy = action.parameters["strategy"]
        options: dict[str, Any] = {"strategy": strategy}
        if strategy == "constant":
            options["fill_value"] = action.parameters["fill_value"]
        return ExecutionArtifact(
            ExecutionArtifactKind.COLUMN_TRANSFORMER,
            SimpleImputer(**options),
        )


class ScalingFactory(TransformationFactory):
    name = "scale_numeric"

    def build(self, action: TransformationAction) -> ExecutionArtifact:
        transformer = (
            StandardScaler()
            if action.parameters["method"] == "standard"
            else RobustScaler()
        )
        return ExecutionArtifact(ExecutionArtifactKind.COLUMN_TRANSFORMER, transformer)


class LogTransformFactory(TransformationFactory):
    name = "log_transform"

    def build(self, action: TransformationAction) -> ExecutionArtifact:
        return ExecutionArtifact(ExecutionArtifactKind.COLUMN_TRANSFORMER, SafeLog1pTransformer())


class OneHotEncodingFactory(TransformationFactory):
    name = "one_hot_encode"

    def build(self, action: TransformationAction) -> ExecutionArtifact:
        return ExecutionArtifact(
            ExecutionArtifactKind.COLUMN_TRANSFORMER,
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
        )


class ClassWeightFactory(TransformationFactory):
    name = "class_weight"

    def build(self, action: TransformationAction) -> ExecutionArtifact:
        return ExecutionArtifact(
            ExecutionArtifactKind.ESTIMATOR_PARAMETERS,
            {"class_weight": action.parameters["mode"]},
        )


class SamplingFactory(TransformationFactory):
    name = "resample_classes"

    def build(self, action: TransformationAction) -> ExecutionArtifact:
        method = action.parameters["method"]
        seed = action.parameters["random_seed"]
        samplers = {
            "random_undersampling": RandomUnderSampler(random_state=seed),
            "random_oversampling": RandomOverSampler(random_state=seed),
            "smote": SMOTE(random_state=seed),
        }
        return ExecutionArtifact(ExecutionArtifactKind.SAMPLER, samplers[method])


class TransformationFactoryRegistry:
    """Immutable-by-convention mapping from safe action names to factories."""

    def __init__(self, factories: Iterable[TransformationFactory] = ()) -> None:
        self._factories: dict[str, TransformationFactory] = {}
        for factory in factories:
            self.register(factory)

    def register(self, factory: TransformationFactory) -> None:
        if not factory.name:
            raise ValueError("Transformation factory name cannot be empty")
        if factory.name in self._factories:
            raise ValueError(f"Transformation factory already registered: {factory.name}")
        self._factories[factory.name] = factory

    def require(self, name: str) -> TransformationFactory:
        """Resolve a factory or fail closed if execution support is absent."""

        factory = self._factories.get(name)
        if factory is None:
            raise ValueError(f"No executable factory is registered for '{name}'")
        return factory

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._factories)

    @classmethod
    def default(cls) -> "TransformationFactoryRegistry":
        return cls(
            [
                NumericImputationFactory(),
                CategoricalImputationFactory(),
                ScalingFactory(),
                LogTransformFactory(),
                OneHotEncodingFactory(),
                ClassWeightFactory(),
                SamplingFactory(),
            ]
        )


class ModelPipelineBuilder:
    """Compose model-specific pipelines from trusted execution artifacts."""

    def __init__(self, factory_registry: TransformationFactoryRegistry) -> None:
        self._factory_registry = factory_registry

    @property
    def supported_transformations(self) -> frozenset[str]:
        return self._factory_registry.names

    def build(
        self,
        plan: TransformationPlan,
        *,
        model_name: str,
        estimator: ClassifierMixin,
    ) -> tuple[ImbalancedPipeline, tuple[AppliedAction, ...]]:
        """Build one pipeline while preserving plan order for every column."""

        column_steps: dict[str, list[tuple[str, TransformerMixin]]] = {}
        sampler: Any | None = None
        estimator_parameters: dict[str, Any] = {}
        applied: list[AppliedAction] = []

        for order, action in enumerate(plan.actions):
            if action.model_scope not in {"all", model_name}:
                continue
            factory = self._factory_registry.require(action.transformation)
            artifact = factory.build(action)

            if artifact.kind == ExecutionArtifactKind.COLUMN_TRANSFORMER:
                for column in action.columns:
                    # sklearn estimators are cloned so a multi-column action never
                    # shares fitted state between otherwise independent columns.
                    step_name = self._safe_step_name(order, action.transformation)
                    column_steps.setdefault(column, []).append((step_name, clone(artifact.value)))
            elif artifact.kind == ExecutionArtifactKind.SAMPLER:
                if sampler is not None:
                    raise ValueError("A model pipeline may contain at most one class sampler")
                sampler = artifact.value
            elif artifact.kind == ExecutionArtifactKind.ESTIMATOR_PARAMETERS:
                estimator_parameters.update(artifact.value)
            else:  # pragma: no cover - protects against future incomplete enum handling.
                raise ValueError(f"Unsupported execution artifact: {artifact.kind}")

            applied.append(
                AppliedAction(
                    action_id=action.action_id,
                    transformation=action.transformation,
                    columns=action.columns,
                    model_scope=action.model_scope,
                )
            )

        fitted_estimator = clone(estimator)
        if estimator_parameters:
            available = fitted_estimator.get_params(deep=False)
            unsupported = sorted(set(estimator_parameters).difference(available))
            if unsupported:
                raise ValueError(
                    f"Estimator '{model_name}' does not support parameters: {', '.join(unsupported)}"
                )
            fitted_estimator.set_params(**estimator_parameters)

        pipeline_steps: list[tuple[str, Any]] = []
        if column_steps:
            transformers = [
                (f"column_{index}", Pipeline(steps), [column])
                for index, (column, steps) in enumerate(column_steps.items())
            ]
            pipeline_steps.append(
                (
                    "preprocessing",
                    ColumnTransformer(
                        transformers=transformers,
                        remainder="passthrough",
                        verbose_feature_names_out=False,
                    ),
                )
            )
        if sampler is not None:
            # imblearn.Pipeline invokes fit_resample only during fit. Predict and
            # evaluation data bypass the sampler, preventing validation leakage.
            pipeline_steps.append(("sampler", sampler))
        pipeline_steps.append(("classifier", fitted_estimator))
        return ImbalancedPipeline(pipeline_steps), tuple(applied)

    @staticmethod
    def _safe_step_name(order: int, transformation: str) -> str:
        """Create a deterministic sklearn-compatible step identifier."""

        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", transformation).replace("__", "_")
        return f"step_{order}_{safe_name}"
