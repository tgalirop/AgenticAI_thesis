"""Generate and strictly parse model-proposed transformation plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from agenticai_thesis.agentic.domain import TransformationPlan
from agenticai_thesis.agentic.model_clients import ModelClientProtocol
from agenticai_thesis.agentic.prompts import StrategyPromptContext, StrategyPromptProvider


class TransformationPlanParseError(ValueError):
    """Raised when model output cannot cross the typed plan boundary."""


class TransformationPlanParser:
    """Convert an untrusted JSON object into an immutable domain model."""

    def parse(self, payload: Mapping[str, Any]) -> TransformationPlan:
        """Reject missing, malformed, or additional plan fields via Pydantic."""

        try:
            return TransformationPlan.model_validate(dict(payload))
        except ValidationError as exc:
            raise TransformationPlanParseError(
                f"Model output is not a valid TransformationPlan: {exc}"
            ) from exc


class StrategyGenerator:
    """Provider-neutral application service for one strategy proposal."""

    def __init__(
        self,
        model_client: ModelClientProtocol,
        prompt_provider: StrategyPromptProvider,
        parser: TransformationPlanParser,
        allowed_transformations: Sequence[str],
    ) -> None:
        if not isinstance(model_client, ModelClientProtocol):
            raise TypeError("model_client must satisfy ModelClientProtocol")
        if not allowed_transformations:
            raise ValueError("At least one transformation must be allowed")
        self._model_client = model_client
        self._prompt_provider = prompt_provider
        self._parser = parser
        self._allowed_transformations = tuple(sorted(set(allowed_transformations)))

    def generate(self, context: StrategyPromptContext) -> TransformationPlan:
        """Generate a plan, validate its schema, and verify request identity."""

        system_prompt, user_prompt = self._prompt_provider.build(
            context,
            allowed_transformations=self._allowed_transformations,
        )
        payload = self._model_client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=TransformationPlan.model_json_schema(),
        )
        plan = self._parser.parse(payload)

        # Schema validation alone cannot establish that the model answered for
        # the requested dataset/iteration, so enforce both invariants here.
        if plan.dataset_id != context.dataset.dataset_id:
            raise TransformationPlanParseError("Model plan uses an unexpected dataset_id")
        if plan.iteration != context.iteration:
            raise TransformationPlanParseError("Model plan uses an unexpected iteration")
        return plan
