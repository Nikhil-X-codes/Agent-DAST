"""Live integration tests for Phase 8 automatic HTML/JSON report generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

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


def test_live_scan_automatically_generates_html_report(tmp_path: Path) -> None:
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
        agent = CustomerSupportAgent(router, db, variant="hardened", authenticated_customer_id="cust_alice")
        harness = LangGraphHarness(agent, TargetConfig(target_name="live-hardened", authorized=True))
        orchestrator = ScanOrchestrator(
            harness,
            policy,
            budget=ScanBudget(max_tests=2, max_turns_per_test=2, max_total_tokens=15_000),
            trace_store=store,
        )
        orchestrator.run(scan_id="live-rep-001", report_dir=tmp_path)

    html_file = tmp_path / "live-rep-001.html"
    json_file = tmp_path / "live-rep-001.json"
    assert html_file.exists()
    assert json_file.exists()

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["completed_tests"] == 2
    assert data["total_tokens_used"] > 0
    assert len(data["providers_used"]) >= 1
