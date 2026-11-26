"""Live integration tests for Phase 4 runtime execution tracing."""

from __future__ import annotations

import os

from dotenv import load_dotenv
import pytest

from agentshield.harnesses import LangGraphHarness, SessionContext, TargetConfig
from agentshield.providers import FreeProviderRouter
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB
from agentshield.tracing import TraceCollector, TraceStore

load_dotenv(override=False)

pytestmark = pytest.mark.integration


def _has_any_key() -> bool:
    return any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"))


def test_live_trace_collector_persists_real_execution_evidence() -> None:
    if not _has_any_key():
        pytest.skip("No free-tier provider API keys set in environment")

    with (
        FreeProviderRouter.from_env() as router,
        SupportSandboxDB() as db,
        TraceStore(":memory:") as store,
    ):
        agent = CustomerSupportAgent(
            router,
            db,
            variant="hardened",
            authenticated_customer_id="cust_alice",
        )
        harness = LangGraphHarness(agent, TargetConfig(target_name="live-support", authorized=True))
        collector = TraceCollector(store)
        session = SessionContext(authenticated_subject="cust_alice")

        result = collector.trace_execution(
            harness,
            "Please refund my order ord_1001 because the keyboard arrived with a broken key.",
            scan_id="live-scan-001",
            session=session,
        )

        llm_events = store.get_llm_events("live-scan-001", session_id=session.session_id)
        tool_events = store.get_tool_events("live-scan-001", session_id=session.session_id)

        assert result.response_text
        assert len(llm_events) >= 1
        assert any(ev.tool_name == "request_refund" and ev.state_changed is True for ev in tool_events)
        assert all(ev.authorization == "allowed" for ev in tool_events)
