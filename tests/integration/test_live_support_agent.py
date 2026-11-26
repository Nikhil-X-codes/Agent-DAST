"""Live integration tests for the LangGraph + SQLite CustomerSupportAgent."""

from __future__ import annotations

import os

from dotenv import load_dotenv
import pytest

from agentshield.providers import FreeProviderRouter
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB

load_dotenv(override=False)

pytestmark = pytest.mark.integration


def _has_any_key() -> bool:
    return any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"))


def test_live_agent_selects_and_executes_get_order() -> None:
    if not _has_any_key():
        pytest.skip("No free-tier provider API keys set in environment")

    with FreeProviderRouter.from_env() as router, SupportSandboxDB() as db:
        agent = CustomerSupportAgent(
            router,
            db,
            variant="hardened",
            authenticated_customer_id="cust_alice",
        )
        result = agent.run("Look up my order ord_1001 and tell me the item name and status.")

    assert result.final_response
    assert result.latency_ms > 0
    assert result.usage.total_tokens > 0
    assert any(rec.tool_name == "get_order" and rec.status == "success" for rec in result.tool_executions)


def test_live_agent_creates_ticket_in_sqlite() -> None:
    if not _has_any_key():
        pytest.skip("No free-tier provider API keys set in environment")

    with FreeProviderRouter.from_env() as router, SupportSandboxDB() as db:
        agent = CustomerSupportAgent(
            router,
            db,
            variant="hardened",
            authenticated_customer_id="cust_alice",
        )
        result = agent.run(
            "Create a support ticket for customer cust_alice with subject 'Broken keycap' "
            "and description 'My Mechanical Keyboard arrived with a cracked keycap.'"
        )
        tickets = db.list_tickets_for_customer("cust_alice")

    assert result.final_response
    assert any(rec.tool_name == "create_ticket" and rec.status == "success" for rec in result.tool_executions)
    assert len(tickets) == 1
