"""Live integration tests for the Phase 9 RAGAgent with real LLM inference."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from agentshield.providers import FreeProviderRouter
from agentshield.targets.rag import RAGAgent, RAGMemoryStore

load_dotenv(override=False)

pytestmark = pytest.mark.integration


def _has_any_key() -> bool:
    return any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"))


def test_live_rag_agent_searches_own_documents() -> None:
    if not _has_any_key():
        pytest.skip("No free-tier provider API keys set in environment")

    with FreeProviderRouter.from_env() as router, RAGMemoryStore() as store:
        agent = RAGAgent(
            router,
            store,
            variant="hardened",
            authenticated_customer_id="cust_alice",
            authenticated_session_id="sess_alice_main",
        )
        result = agent.run("Search documents for my billing onboarding details and order ord_1001 warranty.")

    assert result.final_response
    assert result.usage.total_tokens > 0
    assert any(rec.tool_name == "search_documents" and rec.status == "success" for rec in result.tool_executions)
