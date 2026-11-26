"""Live integration tests for free-tier LLM providers.

These tests make real network calls when the corresponding API key environment
variables are set, and skip cleanly otherwise.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
import pytest

load_dotenv(override=False)

from agentshield.providers import (
    FreeProviderRouter,
    GeminiProvider,
    GroqProvider,
    LLMRequest,
    Message,
    OpenRouterProvider,
)

pytestmark = pytest.mark.integration


def _ping_request() -> LLMRequest:
    return LLMRequest(
        messages=[Message(role="user", content="Reply with the single word: READY")],
        temperature=0.0,
        max_tokens=256,
    )


def test_live_groq_inference() -> None:
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    with GroqProvider() as provider:
        resp = provider.generate(_ping_request())
    assert resp.provider == "groq"
    assert resp.content and len(resp.content.strip()) > 0
    assert resp.latency_ms > 0
    assert resp.usage.total_tokens > 0


def test_live_gemini_inference() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    with GeminiProvider() as provider:
        resp = provider.generate(_ping_request())
    assert resp.provider == "gemini"
    assert resp.content and len(resp.content.strip()) > 0
    assert resp.latency_ms > 0
    assert resp.usage.total_tokens > 0


def test_live_openrouter_inference() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
    with OpenRouterProvider() as provider:
        resp = provider.generate(_ping_request())
    assert resp.provider == "openrouter"
    assert resp.content and len(resp.content.strip()) > 0
    assert resp.latency_ms > 0


def test_live_router_from_env() -> None:
    if not any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")):
        pytest.skip("No free-tier provider API keys set in environment")
    with FreeProviderRouter.from_env() as router:
        resp = router.generate(_ping_request())
    assert resp.content and len(resp.content.strip()) > 0
    assert resp.latency_ms > 0
