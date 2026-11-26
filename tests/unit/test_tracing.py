"""Unit tests for Phase 4 runtime execution tracing, SQLite persistence, and redaction."""

from __future__ import annotations

import json

import httpx
import pytest

from agentshield.harnesses import (
    LangGraphHarness,
    PythonCallableHarness,
    SessionContext,
    TargetConfig,
    TargetExecutionError,
)
from agentshield.providers import GroqProvider, ProviderConfig
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB
from agentshield.tracing import TraceCollector, TraceStore, redact_sensitive

pytestmark = pytest.mark.unit


def _make_sequence_provider(responses: list[dict[str, object]]) -> GroqProvider:
    idx = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal idx
        payload = responses[idx]
        idx += 1
        return httpx.Response(200, json=payload)

    return GroqProvider(
        ProviderConfig(api_key="gsk_secret_unit_test_key_1234567890", model="openai/gpt-oss-20b", max_retries=0),
        transport=httpx.MockTransport(handler),
    )


def test_redactor_masks_credentials_and_preserves_token_counts() -> None:
    payload = {
        "api_key": "secret-123",
        "headers": {"Authorization": "Bearer my-jwt-token", "x-goog-api-key": "AIzaSyDummySecretValue123456"},
        "prompt": "Use key gsk_AbCdEfGhIjKlMnOpQrStUvWxYz to authenticate",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    redacted = redact_sensitive(payload)
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["headers"]["Authorization"] == "[REDACTED]"
    assert redacted["headers"]["x-goog-api-key"] == "[REDACTED]"
    assert "gsk_" not in redacted["prompt"]
    assert "[REDACTED]" in redacted["prompt"]
    assert redacted["usage"]["total_tokens"] == 15


def test_trace_collector_persists_state_changes_and_missing_authz() -> None:
    provider = _make_sequence_provider(
        [
            {
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_refund_1",
                                    "type": "function",
                                    "function": {
                                        "name": "request_refund",
                                        "arguments": json.dumps(
                                            {"order_id": "ord_2001", "reason": "gsk_AbCdEfGhIjKlMnOpQrStUvWxYz"}
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
            },
            {
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {"message": {"role": "assistant", "content": "Refund processed."}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 45, "completion_tokens": 5, "total_tokens": 50},
            },
        ]
    )

    with SupportSandboxDB(":memory:") as db, TraceStore(":memory:") as store:
        agent = CustomerSupportAgent(provider, db, variant="vulnerable", authenticated_customer_id="cust_alice")
        harness = LangGraphHarness(agent, TargetConfig(target_name="vuln-support", authorized=True))
        collector = TraceCollector(store)
        session = SessionContext(session_id="sess-001", authenticated_subject="cust_alice")

        collector.trace_execution(harness, "Refund ord_2001", scan_id="scan-001", session=session)

        saved_session = store.get_session("scan-001")
        assert saved_session is not None
        assert saved_session.telemetry_available is True

        llm_events = store.get_llm_events("scan-001", session_id="sess-001")
        assert len(llm_events) == 2
        assert llm_events[0].usage is not None
        assert llm_events[0].usage.total_tokens == 35

        tool_events = store.get_tool_events("scan-001", session_id="sess-001")
        assert len(tool_events) == 1
        ev = tool_events[0]
        assert ev.tool_call_id == "call_refund_1"
        assert ev.tool_name == "request_refund"
        assert ev.authorization == "missing"
        assert ev.execution_status == "completed"
        assert ev.state_changed is True
        assert "gsk_" not in json.dumps(ev.arguments)
        assert "[REDACTED]" in ev.arguments["reason"]


def test_trace_collector_records_denied_hardened_without_state_change() -> None:
    provider = _make_sequence_provider(
        [
            {
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_get_1",
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
                "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
            },
            {
                "model": "openai/gpt-oss-20b",
                "choices": [
                    {"message": {"role": "assistant", "content": "Access denied."}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
            },
        ]
    )

    with SupportSandboxDB(":memory:") as db, TraceStore(":memory:") as store:
        agent = CustomerSupportAgent(provider, db, variant="hardened", authenticated_customer_id="cust_alice")
        harness = LangGraphHarness(agent, TargetConfig(target_name="hardened-support", authorized=True))
        collector = TraceCollector(store)

        collector.trace_execution(harness, "Show ord_2001", scan_id="scan-002")

        tool_events = store.get_tool_events("scan-002")
        assert len(tool_events) == 1
        assert tool_events[0].authorization == "denied"
        assert tool_events[0].execution_status == "denied"
        assert tool_events[0].state_changed is False


def test_trace_collector_black_box_does_not_fabricate_tool_events() -> None:
    harness = PythonCallableHarness(
        lambda prompt: f"Reply: {prompt}",
        TargetConfig(target_name="bb-py", authorized=True),
        observability="black_box",
    )
    with TraceStore(":memory:") as store:
        collector = TraceCollector(store)
        collector.trace_execution(harness, "Hello", scan_id="scan-bb")

        session = store.get_session("scan-bb")
        assert session is not None
        assert session.telemetry_available is False
        assert len(store.get_llm_events("scan-bb")) == 1
        assert store.get_tool_events("scan-bb") == []


def test_trace_collector_records_errors_on_failure() -> None:
    def failing_agent(_: str) -> str:
        raise RuntimeError("boom")

    harness = PythonCallableHarness(failing_agent, TargetConfig(target_name="fail-py", authorized=True))
    with TraceStore(":memory:") as store:
        collector = TraceCollector(store)
        with pytest.raises(TargetExecutionError):
            collector.trace_execution(harness, "Trigger", scan_id="scan-err")

        events = store.get_llm_events("scan-err")
        assert len(events) == 1
        assert events[0].error is not None
        assert "boom" in events[0].error
