"""Unit tests for provider adapters without making external API calls."""

import json
from unittest.mock import patch

import pytest

from agenticai_thesis.agentic.model_clients import (
    GroqClientConfig,
    GroqModelClient,
    ModelClientError,
)


class _HttpResponse:
    """Minimal context-managed HTTP response used by the mocked transport."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_HttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_groq_client_requires_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = GroqModelClient(GroqClientConfig())

    with pytest.raises(ModelClientError, match="Missing GROQ_API_KEY"):
        client.generate_structured(system_prompt="system", user_prompt="user", json_schema={})


def test_groq_client_decodes_structured_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-only-secret")
    response = _HttpResponse(
        {"choices": [{"message": {"content": '{"plan_id": "plan_001"}'}}]}
    )

    with patch("agenticai_thesis.agentic.model_clients.request.urlopen", return_value=response) as call:
        result = GroqModelClient(GroqClientConfig()).generate_structured(
            system_prompt="system",
            user_prompt="user",
            json_schema={"type": "object"},
        )

    assert result == {"plan_id": "plan_001"}
    sent_request = call.call_args.args[0]
    sent_payload = json.loads(sent_request.data.decode("utf-8"))
    assert sent_payload["model"] == "openai/gpt-oss-20b"
    assert sent_payload["response_format"]["json_schema"]["strict"] is False
    assert sent_request.headers["Authorization"] == "Bearer test-only-secret"
    assert sent_request.headers["User-agent"] == "agenticai-thesis/0.1.0"
