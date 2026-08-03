"""Replaceable clients for language models used by the Agentic workflow.

The rest of the project depends only on :class:`ModelClientProtocol`.  This
keeps LangGraph nodes independent of any provider and makes every unit test fully
local, deterministic, and free of model inference costs.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib import error, request


JsonObject = dict[str, Any]


class ModelClientError(RuntimeError):
    """Raised when a model endpoint fails or returns an invalid envelope."""


@runtime_checkable
class ModelClientProtocol(Protocol):
    """Small provider-neutral contract required by the Strategy Generator."""

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Mapping[str, Any],
    ) -> JsonObject:
        """Return one JSON object conforming to the requested schema."""


@dataclass(frozen=True, slots=True)
class GroqClientConfig:
    """Safe Groq Free Tier settings without storing the secret API key."""

    model: str = "openai/gpt-oss-20b"
    base_url: str = "https://api.groq.com/openai/v1"
    api_key_environment_variable: str = "GROQ_API_KEY"
    timeout_seconds: float = 120.0
    temperature: float = 0.0
    reasoning_effort: str = "medium"
    max_rate_limit_retries: int = 2

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Groq model must not be empty")
        if not self.base_url.startswith("https://"):
            raise ValueError("Groq base_url must use HTTPS")
        if not self.api_key_environment_variable.strip():
            raise ValueError("Groq API-key environment-variable name must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("Groq timeout_seconds must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Groq temperature must be between 0 and 2")
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("Groq reasoning_effort must be low, medium, or high")
        if self.max_rate_limit_retries < 0:
            raise ValueError("Groq max_rate_limit_retries cannot be negative")


class GroqModelClient:
    """Use GPT-OSS through Groq's OpenAI-compatible Chat Completions API.

    No billing information is handled by this class.  A Free Tier account can
    call the same endpoint; when its quota is exhausted Groq returns HTTP 429.
    The key is read at call time from the environment and is never serialized,
    logged, stored in Agent state, or committed to the repository.
    """

    def __init__(self, config: GroqClientConfig) -> None:
        self._config = config

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Mapping[str, Any],
    ) -> JsonObject:
        """Request schema-guided JSON and retry only Free Tier rate limits."""

        api_key = os.environ.get(self._config.api_key_environment_variable)
        if not api_key:
            raise ModelClientError(
                f"Missing {self._config.api_key_environment_variable}. "
                "Create a Groq Free Tier key and set it as an environment variable."
            )

        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "temperature": self._config.temperature,
            "reasoning_effort": self._config.reasoning_effort,
            # The current TransformationPlan contains transformation-specific
            # parameter dictionaries. Groq strict mode disallows such open-ended
            # objects, therefore schema mode is best-effort here and our strict
            # Pydantic parser plus allowlist validator remain authoritative.
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "transformation_plan",
                    "strict": False,
                    "schema": dict(json_schema),
                },
            },
        }
        endpoint = f"{self._config.base_url.rstrip('/')}/chat/completions"

        for attempt in range(self._config.max_rate_limit_retries + 1):
            http_request = request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    # A stable application identity prevents edge-security
                    # systems from rejecting Python urllib's default signature.
                    "User-Agent": "agenticai-thesis/0.1.0",
                },
                method="POST",
            )
            try:
                with request.urlopen(http_request, timeout=self._config.timeout_seconds) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
                return self._parse_response(envelope)
            except error.HTTPError as exc:
                if exc.code == 429 and attempt < self._config.max_rate_limit_retries:
                    # Respect the provider hint but cap the wait.  Repeated 429s
                    # fail explicitly instead of looping or moving to a paid tier.
                    retry_after = float(exc.headers.get("Retry-After", "1"))
                    time.sleep(min(max(retry_after, 0.0), 30.0))
                    continue
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429:
                    raise ModelClientError(
                        "Groq Free Tier rate limit reached; wait for the quota reset."
                    ) from exc
                if exc.code in {401, 403}:
                    raise ModelClientError("Groq rejected the configured API key") from exc
                raise ModelClientError(f"Groq returned HTTP {exc.code}: {details}") from exc
            except error.URLError as exc:
                raise ModelClientError("Cannot reach the Groq API") from exc
            except (TimeoutError, json.JSONDecodeError) as exc:
                raise ModelClientError("Groq returned no valid JSON response envelope") from exc

        raise ModelClientError("Groq request failed after rate-limit retries")

    @staticmethod
    def _parse_response(envelope: Any) -> JsonObject:
        """Extract and decode the assistant JSON from the API envelope."""

        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelClientError("Groq response is missing choices[0].message.content") from exc
        if not isinstance(content, str):
            raise ModelClientError("Groq message content must be a JSON string")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelClientError("Groq message content is not valid JSON") from exc
        if not isinstance(result, dict):
            raise ModelClientError("Groq structured output must be a JSON object")
        return result


@dataclass(frozen=True, slots=True)
class OllamaClientConfig:
    """Connection and inference settings for a local Ollama server."""

    model: str = "gpt-oss:20b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 180.0
    temperature: float = 0.0
    num_ctx: int = 16_384

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Ollama model must not be empty")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("Ollama base_url must be an HTTP(S) URL")
        if self.timeout_seconds <= 0:
            raise ValueError("Ollama timeout_seconds must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Ollama temperature must be between 0 and 2")
        if self.num_ctx < 1:
            raise ValueError("Ollama num_ctx must be positive")


class OllamaModelClient:
    """Call Ollama's local chat endpoint and require schema-constrained JSON.

    The implementation intentionally uses Python's standard HTTP library.  It
    avoids coupling the research code to a particular LangChain/Ollama wrapper,
    while the constructor keeps the transport replaceable for focused tests.
    """

    def __init__(self, config: OllamaClientConfig) -> None:
        self._config = config

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Mapping[str, Any],
    ) -> JsonObject:
        """Request a non-streaming response constrained by a JSON Schema."""

        payload = {
            "model": self._config.model,
            "stream": False,
            # Ollama accepts a full JSON Schema in ``format``.  This is stronger
            # than merely asking for JSON in natural-language instructions.
            "format": dict(json_schema),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self._config.temperature,
                "num_ctx": self._config.num_ctx,
            },
        }
        endpoint = f"{self._config.base_url.rstrip('/')}/api/chat"
        http_request = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self._config.timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise ModelClientError(f"Ollama returned HTTP {exc.code}: {details}") from exc
        except error.URLError as exc:
            raise ModelClientError(
                "Cannot reach Ollama. Start it locally and confirm the configured base_url."
            ) from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError("Ollama returned no valid JSON response envelope") from exc

        try:
            content = envelope["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ModelClientError("Ollama response is missing message.content") from exc
        if not isinstance(content, str):
            raise ModelClientError("Ollama message.content must be a JSON string")

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelClientError("Ollama content is not valid JSON") from exc
        if not isinstance(result, dict):
            raise ModelClientError("Ollama structured output must be a JSON object")
        return result


@dataclass(slots=True)
class FakeModelClient:
    """Deterministic queued model responses for tests and offline demos."""

    responses: deque[JsonObject] = field(default_factory=deque)
    calls: list[JsonObject] = field(default_factory=list, init=False)

    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        # Defensive copies prevent a test from mutating a queued answer after
        # constructing the fake client.
        self.responses = deque(dict(response) for response in responses)
        self.calls = []

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Mapping[str, Any],
    ) -> JsonObject:
        """Record the request and return the next predefined response."""

        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "json_schema": dict(json_schema),
            }
        )
        if not self.responses:
            raise ModelClientError("FakeModelClient has no queued response")
        return dict(self.responses.popleft())
