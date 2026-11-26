"""Unit tests for Groq, Gemini, and OpenRouter provider adapters."""

from __future__ import annotations

import json

import httpx
import pytest

from agentshield.providers import (
    GeminiProvider,
    GroqProvider,
    LLMRequest,
    Message,
    OpenRouterProvider,
    ProviderConfig,
    ProviderConfigError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    ToolDefinition,
    ZeroCostViolationError,
)

pytestmark = pytest.mark.unit


def _basic_request(**kwargs: object) -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", content="Say hello")], **kwargs)


def test_zero_cost_blocks_paid_models() -> None:
    groq = GroqProvider(ProviderConfig(api_key="test-key", model="llama-3.1-8b-instant"))
    with pytest.raises(ZeroCostViolationError):
        groq.generate(_basic_request(model="gpt-4o"))

    gemini = GeminiProvider(ProviderConfig(api_key="test-key", model="gemini-2.5-flash"))
    with pytest.raises(ZeroCostViolationError):
        gemini.generate(_basic_request(model="gemini-2.5-pro"))

    openrouter = OpenRouterProvider(
        ProviderConfig(api_key="test-key", model="meta-llama/llama-3.3-70b-instruct:free")
    )
    with pytest.raises(ZeroCostViolationError):
        openrouter.generate(_basic_request(model="openai/gpt-4o"))


def test_missing_api_key_raises_config_error() -> None:
    provider = GroqProvider(ProviderConfig(api_key="", model="llama-3.1-8b-instant"))
    with pytest.raises(ProviderConfigError, match="GROQ_API_KEY"):
        provider.generate(_basic_request())


def test_api_key_not_leaked_in_repr_or_error() -> None:
    secret = "super-secret-api-key-12345"
    cfg = ProviderConfig(api_key=secret, model="llama-3.1-8b-instant", max_retries=0)
    assert secret not in repr(cfg)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Unauthorized"}})

    provider = GroqProvider(cfg, transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as exc_info:
        provider.generate(_basic_request())
    assert secret not in str(exc_info.value)


def test_groq_inference_structured_and_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-groq-key"
        body = json.loads(request.content)
        assert body["model"] == "llama-3.1-8b-instant"
        assert body["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "model": "llama-3.1-8b-instant",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"status": "safe"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
            },
        )

    provider = GroqProvider(
        ProviderConfig(api_key="test-groq-key", model="llama-3.1-8b-instant"),
        transport=httpx.MockTransport(handler),
    )
    resp = provider.generate(
        _basic_request(
            response_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": ["status"],
            }
        )
    )
    assert resp.provider == "groq"
    assert resp.model == "llama-3.1-8b-instant"
    assert resp.parsed == {"status": "safe"}
    assert resp.usage.total_tokens == 18
    assert resp.latency_ms >= 0.0
    assert provider.get_usage_summary()["total_tokens"] == 18


def test_openrouter_tool_calling() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "read_db"
        return httpx.Response(
            200,
            json={
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_db",
                                        "arguments": '{"table": "users"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        )

    provider = OpenRouterProvider(
        ProviderConfig(api_key="test-or-key", model="meta-llama/llama-3.3-70b-instruct:free"),
        transport=httpx.MockTransport(handler),
    )
    resp = provider.generate(
        _basic_request(
            tools=[
                ToolDefinition(
                    name="read_db",
                    description="Read synthetic table",
                    parameters={"type": "object", "properties": {"table": {"type": "string"}}},
                )
            ]
        )
    )
    assert resp.provider == "openrouter"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "read_db"
    assert resp.tool_calls[0].arguments == {"table": "users"}
    assert resp.usage.total_tokens == 30


def test_gemini_inference_and_tool_calling() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
        assert "key=" not in str(request.url)
        assert request.headers["x-goog-api-key"] == "test-gemini-key"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Calling tool"},
                                {"functionCall": {"name": "check_policy", "args": {"id": 7}}},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 15,
                    "candidatesTokenCount": 9,
                    "totalTokenCount": 24,
                },
            },
        )

    provider = GeminiProvider(
        ProviderConfig(api_key="test-gemini-key", model="gemini-2.5-flash"),
        transport=httpx.MockTransport(handler),
    )
    resp = provider.generate(_basic_request())
    assert resp.provider == "gemini"
    assert resp.content == "Calling tool"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "check_policy"
    assert resp.tool_calls[0].arguments == {"id": 7}
    assert resp.usage.total_tokens == 24


def test_rate_limit_retries_then_succeeds() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            json={
                "model": "llama-3.1-8b-instant",
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    provider = GroqProvider(
        ProviderConfig(
            api_key="test-key",
            model="llama-3.1-8b-instant",
            max_retries=2,
            retry_backoff_seconds=0.0,
        ),
        transport=httpx.MockTransport(handler),
    )
    resp = provider.generate(_basic_request())
    assert attempts == 2
    assert resp.content == "ok"


def test_rate_limit_exhausted_raises_without_mock_fallback() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    provider = GroqProvider(
        ProviderConfig(
            api_key="test-key",
            model="llama-3.1-8b-instant",
            max_retries=1,
            retry_backoff_seconds=0.0,
        ),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(RateLimitError):
        provider.generate(_basic_request())


def test_timeout_raises_provider_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = GroqProvider(
        ProviderConfig(
            api_key="test-key",
            model="llama-3.1-8b-instant",
            max_retries=1,
            retry_backoff_seconds=0.0,
        ),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderTimeoutError):
        provider.generate(_basic_request())
