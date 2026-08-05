"""Unit tests for provider adapters without making external API calls."""

import json
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError

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
        {
            "choices": [{"message": {"content": '{"plan_id": "plan_001"}'}}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "prompt_time": 0.25,
                "completion_time": 0.1,
                "total_time": 0.35,
            },
        }
    )

    client = GroqModelClient(GroqClientConfig())
    with patch("agenticai_thesis.agentic.model_clients.request.urlopen", return_value=response) as call:
        result = client.generate_structured(
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
    usage = client.usage_summary
    assert usage.api_requests == 1
    assert usage.successful_responses == 1
    assert usage.failed_requests == 0
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 30
    assert usage.total_tokens == 150
    assert usage.provider_total_seconds == pytest.approx(0.35)


def _http_error(code: int, payload: dict[str, object]) -> HTTPError:
    """Construct a realistic urllib error with a readable JSON response body."""

    return HTTPError(
        url="https://api.groq.com/openai/v1/chat/completions",
        code=code,
        msg="test error",
        hdrs={},
        fp=BytesIO(json.dumps(payload).encode("utf-8")),
    )


def test_groq_client_retries_provider_json_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-only-secret")
    validation_error = _http_error(
        400,
        {"error": {"code": "json_validate_failed", "message": "invalid model output"}},
    )
    valid_response = _HttpResponse(
        {"choices": [{"message": {"content": '{"plan_id": "plan_002"}'}}]}
    )

    with patch(
        "agenticai_thesis.agentic.model_clients.request.urlopen",
        side_effect=[validation_error, valid_response],
    ) as call:
        client = GroqModelClient(GroqClientConfig(max_rate_limit_retries=1))
        result = client.generate_structured(
            system_prompt="system", user_prompt="user", json_schema={"type": "object"}
        )

    assert result == {"plan_id": "plan_002"}
    assert call.call_count == 2
    assert client.usage_summary.api_requests == 2
    assert client.usage_summary.failed_requests == 1
    assert client.usage_summary.successful_responses == 1
    assert client.usage_summary.retries == 1


def test_groq_client_does_not_retry_unrelated_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-only-secret")
    bad_request = _http_error(400, {"error": {"code": "invalid_request"}})

    with patch(
        "agenticai_thesis.agentic.model_clients.request.urlopen",
        side_effect=bad_request,
    ) as call:
        with pytest.raises(ModelClientError, match="HTTP 400"):
            GroqModelClient(GroqClientConfig(max_rate_limit_retries=2)).generate_structured(
                system_prompt="system", user_prompt="user", json_schema={}
            )

    assert call.call_count == 1
