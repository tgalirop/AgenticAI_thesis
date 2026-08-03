"""Prompt construction isolated from model I/O and orchestration logic."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agenticai_thesis.agentic.domain import DatasetContext


# Prompt guidance mirrors the deterministic registry contracts.  It improves
# first-pass validity but never replaces validator enforcement.  Keeping the
# catalog injectable allows future transformation plugins to supply contracts
# without changing Strategy Generator control flow.
DEFAULT_TRANSFORMATION_CONTRACTS: dict[str, dict[str, Any]] = {
    "impute_numeric": {
        "columns": "one or more numeric columns",
        "parameters": {"strategy": ["median", "mean"]},
    },
    "impute_categorical": {
        "columns": "one or more categorical columns",
        "parameters": {
            "strategy": ["most_frequent", "constant"],
            "fill_value": "required only when strategy=constant",
        },
    },
    "scale_numeric": {
        "columns": "one or more numeric columns",
        "parameters": {"method": ["standard", "robust"]},
    },
    "log_transform": {
        "columns": "one or more known non-negative numeric columns",
        "parameters": {},
    },
    "one_hot_encode": {
        "columns": "one or more categorical columns",
        "parameters": {"handle_unknown": "ignore"},
    },
    "class_weight": {
        "columns": [],
        "parameters": {"mode": "balanced"},
    },
    "resample_classes": {
        "columns": [],
        "parameters": {
            "method": ["random_undersampling", "random_oversampling", "smote"],
            "random_seed": "integer",
        },
    },
}


@dataclass(frozen=True, slots=True)
class StrategyPromptContext:
    """Only the compact, safe evidence exposed to the strategy model."""

    dataset: DatasetContext
    iteration: int
    quality_metrics: Mapping[str, Any]
    baseline_metrics: Mapping[str, Any]
    previous_feedback: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.iteration < 1:
            raise ValueError("Strategy iteration must be at least one")


class StrategyPromptProvider:
    """Build stable prompts for auditable and reproducible experiments."""

    SYSTEM_PROMPT = """You are the Strategy Generator of a fraud-detection research agent.
Return exactly one transformation plan matching the supplied JSON Schema.
Use only the explicitly allowed declarative transformations and known columns.
Never emit Python, shell commands, SQL, file paths, or executable code.
Never request access to the held-out temporal test set.
Prefer the smallest defensible plan and explain every action in its rationale.
Select an action only when a supplied quality metric or baseline-pipeline contract
directly justifies it. Do not change class-imbalance handling merely because the
fraud target is imbalanced; assume the baseline estimators already handle it."""

    def __init__(
        self,
        transformation_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        source = transformation_contracts or DEFAULT_TRANSFORMATION_CONTRACTS
        self._contracts = {name: dict(contract) for name, contract in source.items()}

    def build(
        self,
        context: StrategyPromptContext,
        *,
        allowed_transformations: Sequence[str],
    ) -> tuple[str, str]:
        """Return system/user prompts without embedding raw transaction rows."""

        missing_contracts = sorted(set(allowed_transformations).difference(self._contracts))
        if missing_contracts:
            raise ValueError(
                "Missing prompt contracts for transformations: " + ", ".join(missing_contracts)
            )
        evidence = {
            "dataset": context.dataset.model_dump(mode="json"),
            "iteration": context.iteration,
            "quality_metrics": dict(context.quality_metrics),
            "baseline_metrics": dict(context.baseline_metrics),
            "previous_feedback": list(context.previous_feedback),
            "allowed_transformations": sorted(set(allowed_transformations)),
            "transformation_contracts": {
                name: self._contracts[name] for name in sorted(set(allowed_transformations))
            },
        }
        user_prompt = (
            "Propose the next preprocessing plan from this compact experiment evidence. "
            "The plan dataset_id and iteration must exactly match the evidence.\n"
            + json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2)
        )
        return self.SYSTEM_PROMPT, user_prompt
