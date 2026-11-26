"""Unit tests for Phase 7 multi-turn LangGraph ScanOrchestrator."""

from __future__ import annotations

import json

import httpx
import pytest

from agentshield.evaluation import EvaluationPolicy
from agentshield.harnesses import LangGraphHarness, TargetConfig
from agentshield.orchestrator import ScanBudget, ScanOrchestrator
from agentshield.providers import GroqProvider, ProviderConfig
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB
from agentshield.tracing import TraceStore

pytestmark = pytest.mark.unit


def _make_tool_then_reply_provider(*, tool_name: str = "get_order", order_id: str = "ord_2001") -> GroqProvider:
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
                                            "name": tool_name,
                                            "arguments": json.dumps({"order_id": order_id}),
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
                "choices": [{"message": {"role": "assistant", "content": "Processed."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 5, "total_tokens": 30},
            },
        )

    return GroqProvider(
        ProviderConfig(api_key="test-key", model="openai/gpt-oss-20b", max_retries=0),
        transport=httpx.MockTransport(handler),
    )


def test_orchestrator_runs_multi_turn_scan_and_preserves_evidence() -> None:
    provider = _make_tool_then_reply_provider()
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
            budget=ScanBudget(max_tests=3, max_turns_per_test=2, max_total_tokens=10_000),
            trace_store=store,
        )
        report = orchestrator.run(scan_id="scan-multi-001")

        assert report.scan_id == "scan-multi-001"
        assert report.tests_executed == 3
        assert report.total_turns_executed > report.tests_executed
        assert report.termination_reason == "all_tests_completed"
        assert report.overall_verdict == "confirmed"
        assert report.confirmed_findings_count >= 1
        assert len(store.get_tool_events("scan-multi-001")) >= 1


def test_orchestrator_terminates_when_token_budget_exhausted() -> None:
    provider = _make_tool_then_reply_provider()
    policy = EvaluationPolicy(
        authenticated_subject="cust_alice",
        foreign_identifiers=["ord_2001", "cust_bob"],
    )

    with SupportSandboxDB(":memory:") as db:
        agent = CustomerSupportAgent(provider, db, variant="hardened", authenticated_customer_id="cust_alice")
        harness = LangGraphHarness(agent, TargetConfig(target_name="hardened-support", authorized=True))
        orchestrator = ScanOrchestrator(
            harness,
            policy,
            budget=ScanBudget(max_tests=5, max_turns_per_test=2, max_total_tokens=50),
        )
        report = orchestrator.run(scan_id="scan-budget-001")

        assert report.termination_reason == "token_budget_exhausted"
        assert report.tests_executed < 5
        assert report.total_tokens_used >= 50


def test_orchestrator_hardened_target_has_no_confirmed_violations() -> None:
    provider = _make_tool_then_reply_provider()
    policy = EvaluationPolicy(
        authenticated_subject="cust_alice",
        foreign_identifiers=["ord_2001", "cust_bob"],
    )

    with SupportSandboxDB(":memory:") as db:
        agent = CustomerSupportAgent(provider, db, variant="hardened", authenticated_customer_id="cust_alice")
        harness = LangGraphHarness(agent, TargetConfig(target_name="hardened-support", authorized=True))
        orchestrator = ScanOrchestrator(
            harness,
            policy,
            budget=ScanBudget(max_tests=2, max_turns_per_test=2),
        )
        report = orchestrator.run()

        assert report.tests_executed == 2
        assert report.confirmed_findings_count == 0
        assert report.overall_verdict == "no_violation_observed"
