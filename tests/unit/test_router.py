"""Unit tests for FreeProviderRouter fallback and zero-cost guardrails."""

from __future__ import annotations

import httpx
import pytest

from agentshield.providers import (
    AllProvidersExhaustedError,
    FreeProviderRouter,
    GeminiProvider,
    GroqProvider,
    LLMRequest,
    Message,
    OpenRouterProvider,
    ProviderConfig,
    ProviderConfigError,
    ZeroCostViolationError,
)

pytestmark = pytest.mark.unit


def _request(model: str | None = None) -> LLMRequest:
    return LLMRequest(messages=[Message(role="user", content="Ping")], model=model)


def test_router_falls_back_from_rate_limited_groq_to_gemini() -> None:
    groq_transport = httpx.MockTransport(lambda _: httpx.Response(429))
    gemini_transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "pong from gemini"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3, "totalTokenCount": 7},
            },
        )
    )

    groq = GroqProvider(
        ProviderConfig(api_key="k1", model="llama-3.1-8b-instant", max_retries=0),
        transport=groq_transport,
    )
    gemini = GeminiProvider(
        ProviderConfig(api_key="k2", model="gemini-2.5-flash", max_retries=0),
        transport=gemini_transport,
    )

    router = FreeProviderRouter([groq, gemini], zero_cost_mode=True)
    resp = router.generate(_request())

    assert resp.provider == "gemini"
    assert resp.content == "pong from gemini"
    summary = router.get_usage_summary()
    assert summary["total_tokens"] == 7


def test_router_raises_all_providers_exhausted() -> None:
    fail_transport = httpx.MockTransport(lambda _: httpx.Response(429))
    groq = GroqProvider(
        ProviderConfig(api_key="k1", model="llama-3.1-8b-instant", max_retries=0),
        transport=fail_transport,
    )
    openrouter = OpenRouterProvider(
        ProviderConfig(api_key="k2", model="meta-llama/llama-3.3-70b-instruct:free", max_retries=0),
        transport=fail_transport,
    )

    router = FreeProviderRouter([groq, openrouter], zero_cost_mode=True)
    with pytest.raises(AllProvidersExhaustedError) as exc_info:
        router.generate(_request())
    assert len(exc_info.value.errors) == 2


def test_router_rejects_non_zero_cost_provider() -> None:
    groq = GroqProvider(
        ProviderConfig(api_key="k1", model="llama-3.1-8b-instant", zero_cost_mode=False)
    )
    with pytest.raises(ZeroCostViolationError):
        FreeProviderRouter([groq], zero_cost_mode=True)


def test_router_does_not_fallback_on_zero_cost_violation() -> None:
    gemini_called = False

    def gemini_handler(_: httpx.Request) -> httpx.Response:
        nonlocal gemini_called
        gemini_called = True
        return httpx.Response(200, json={})

    groq = GroqProvider(ProviderConfig(api_key="k1", model="llama-3.1-8b-instant"))
    gemini = GeminiProvider(
        ProviderConfig(api_key="k2", model="gemini-2.5-flash"),
        transport=httpx.MockTransport(gemini_handler),
    )

    router = FreeProviderRouter([groq, gemini], zero_cost_mode=True)
    with pytest.raises(ZeroCostViolationError):
        router.generate(_request(model="gpt-4o"))
    assert gemini_called is False


def test_router_from_env_requires_at_least_one_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ProviderConfigError):
        FreeProviderRouter.from_env(load_env=False)
