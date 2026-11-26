"""Live integration tests for Phase 7 multi-turn LangGraph ScanOrchestrator."""

from __future__ import annotations

import os

from dotenv import load_dotenv
import pytest

from agentshield.evaluation import EvaluationPolicy
from agentshield.harnesses import LangGraphHarness, TargetConfig
from agentshield.orchestrator import ScanBudget, ScanOrchestrator
from agentshield.providers import FreeProviderRouter
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB
from agentshield.tracing import TraceStore

load_dotenv(override=False)

pytestmark = pytest.mark.integration


def _has_any_key() -> bool:
    return any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"))


def test_live_orchestrator_executes_bounded_multi_turn_scan() -> None:
    if not _has_any_key():
        pytest.skip("No free-tier provider API keys set in environment")

    policy = EvaluationPolicy(
        authenticated_subject="cust_alice",
        foreign_identifiers=["ord_2001", "cust_bob"],
    )

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
        harness = LangGraphHarness(agent, TargetConfig(target_name="live-hardened", authorized=True))
        orchestrator = ScanOrchestrator(
            harness,
            policy,
            budget=ScanBudget(max_tests=2, max_turns_per_test=2, max_total_tokens=15_000),
            trace_store=store,
        )
        report = orchestrator.run(scan_id="live-orch-001")

        assert report.tests_executed == 2
        assert report.total_turns_executed >= 2
        assert report.total_tokens_used > 0
        assert report.confirmed_findings_count == 0
        assert report.termination_reason == "all_tests_completed"
        assert len(store.get_llm_events("live-orch-001")) >= 2
