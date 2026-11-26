"""Live integration tests for Phase 6 context-aware security test engine."""

from __future__ import annotations

import os

from dotenv import load_dotenv
import pytest

from agentshield.evaluation import EvaluationPolicy
from agentshield.harnesses import LangGraphHarness, TargetConfig
from agentshield.providers import FreeProviderRouter
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB
from agentshield.test_engine import SecurityTestEngine, TestSuiteConfig

load_dotenv(override=False)

pytestmark = pytest.mark.integration


def _has_any_key() -> bool:
    return any(os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"))


def test_live_context_aware_suite_runs_against_hardened_agent() -> None:
    if not _has_any_key():
        pytest.skip("No free-tier provider API keys set in environment")

    policy = EvaluationPolicy(
        authenticated_subject="cust_alice",
        foreign_identifiers=["ord_2001", "cust_bob"],
    )
    engine = SecurityTestEngine()

    with FreeProviderRouter.from_env() as router, SupportSandboxDB() as db:
        agent = CustomerSupportAgent(
            router,
            db,
            variant="hardened",
            authenticated_customer_id="cust_alice",
        )
        harness = LangGraphHarness(agent, TargetConfig(target_name="hardened-live", authorized=True))
        result = engine.run_suite(
            harness,
            policy,
            config=TestSuiteConfig(max_tests=2, use_llm_variations=False),
        )

    assert len(result.executions) == 2
    assert result.confirmed_findings_count == 0
    assert "unauthorized_tool_execution" in result.plan.selected_categories
