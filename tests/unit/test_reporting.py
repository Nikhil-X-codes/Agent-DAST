"""Unit tests for Phase 8 HTML/JSON/CLI reporting and comparative benchmarking."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from agentshield.evaluation import EvaluationPolicy
from agentshield.harnesses import LangGraphHarness, TargetConfig
from agentshield.orchestrator import ScanBudget, ScanOrchestrator
from agentshield.providers import GroqProvider, ProviderConfig
from agentshield.reporting import (
    BenchmarkRunner,
    build_audit_report,
    render_cli_report,
)
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB
from agentshield.tracing import TraceStore

pytestmark = pytest.mark.unit


def _make_mock_provider() -> GroqProvider:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        last_msg = body["messages"][-1]
        if last_msg["role"] == "user" and ("ord_2001" in (last_msg.get("content") or "") or "refund" in (last_msg.get("content") or "").lower()):
            return httpx.Response(
                200,
                json={
                    "model": "openai/gpt-oss-20b",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"call_{call_count}",
                                        "type": "function",
                                        "function": {
                                            "name": "get_order",
                                            "arguments": json.dumps({"order_id": "ord_2001"}),
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
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-oss-20b",
                "choices": [{"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 5, "total_tokens": 30},
            },
        )

    return GroqProvider(
        ProviderConfig(api_key="test-key", model="openai/gpt-oss-20b", max_retries=0),
        transport=httpx.MockTransport(handler),
    )


def test_scan_auto_generates_standalone_html_and_json_reports(tmp_path: Path) -> None:
    provider = _make_mock_provider()
    policy = EvaluationPolicy(
        authenticated_subject="cust_alice",
        foreign_identifiers=["ord_2001", "cust_bob"],
    )

    with SupportSandboxDB(":memory:") as db, TraceStore(":memory:") as store:
        agent = CustomerSupportAgent(provider, db, variant="vulnerable", authenticated_customer_id="cust_alice")
        harness = LangGraphHarness(agent, TargetConfig(target_name="vuln-support", authorized=True))
        orchestrator = ScanOrchestrator(
            harness,
            policy,
            budget=ScanBudget(max_tests=2, max_turns_per_test=2),
            trace_store=store,
        )
        scan_report = orchestrator.run(scan_id="auto-rep-1", report_dir=tmp_path)
        audit_data = build_audit_report(scan_report, trace_store=store)

    html_path = tmp_path / "auto-rep-1.html"
    json_path = tmp_path / "auto-rep-1.json"
    assert html_path.exists()
    assert json_path.exists()

    html = html_path.read_text(encoding="utf-8")
    assert "AgentShield Security Audit" in html
    assert "vuln-support" in html
    assert "LLM02:2025" in html
    assert "AML.T0024" in html
    assert "Zero-Cost" in html
    assert "<script src=" not in html

    parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed_json["scan_id"] == "auto-rep-1"
    assert parsed_json["confirmed_count"] == audit_data.confirmed_count
    assert parsed_json["total_tokens_used"] > 0

    buf = StringIO()
    render_cli_report(audit_data, console=Console(file=buf, force_terminal=False))
    assert "vuln-support" in buf.getvalue()


def test_benchmark_runner_compares_vulnerable_vs_hardened_across_trials() -> None:
    provider = _make_mock_provider()
    policy = EvaluationPolicy(
        authenticated_subject="cust_alice",
        foreign_identifiers=["ord_2001", "cust_bob"],
    )
    runner = BenchmarkRunner(
        provider,
        policy,
        budget=ScanBudget(max_tests=2, max_turns_per_test=2),
    )
    comp = runner.run(trials=2)

    assert comp.trials == 2
    assert comp.vulnerable.total_tests_executed == 4
    assert comp.hardened.total_tests_executed == 4
    assert comp.vulnerable.total_confirmed_findings >= 2
    assert comp.hardened.total_confirmed_findings == 0
    assert comp.confirmed_reduction_count >= 2
    assert comp.vulnerable.confirmed_violation_rate > comp.hardened.confirmed_violation_rate
