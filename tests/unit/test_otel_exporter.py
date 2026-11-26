"""Unit tests for OTEL exporter (Phase 4 add-on)."""

from __future__ import annotations

import json

import pytest

from agentshield.tracing.otel_exporter import export_scan_traces_to_file, _build_llm_span, _build_tool_span
from agentshield.tracing.store import TraceStore
from agentshield.tracing.models import LLMTraceEvent, ToolTraceEvent
from agentshield.providers.base import TokenUsage


@pytest.fixture()
def populated_store():
    """TraceStore with one LLM event and two tool events."""
    store = TraceStore(":memory:")
    store.ensure_session(
        __import__("agentshield.tracing.models", fromlist=["ScanSessionRecord"]).ScanSessionRecord(
            scan_id="scan_otel_001",
            target_name="support-vulnerable",
            target_type="langgraph",
            observability="instrumented",
            telemetry_available=True,
        )
    )
    llm_ev = LLMTraceEvent(
        event_id="llm_abc001",
        scan_id="scan_otel_001",
        session_id="sess_001",
        step=1,
        provider="groq",
        model="llama-3.1-8b",
        latency_ms=350.0,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        telemetry_available=True,
    )
    store.record_llm_event(llm_ev)

    for i, tool_name in enumerate(["get_order", "request_refund"]):
        tev = ToolTraceEvent(
            event_id=f"tool_abc00{i + 1}",
            scan_id="scan_otel_001",
            session_id="sess_001",
            tool_name=tool_name,
            authorization="allowed" if tool_name == "get_order" else "denied",
            execution_status="completed" if tool_name == "get_order" else "denied",
            latency_ms=12.0 + i * 5,
            state_changed=False,
        )
        store.record_tool_event(tev)

    yield store
    store.close()


@pytest.mark.unit
class TestOtelExporter:
    def test_export_returns_spans(self, populated_store):
        spans = export_scan_traces_to_file(populated_store, "scan_otel_001")
        assert len(spans) == 3  # 1 LLM + 2 tool spans

    def test_llm_span_has_correct_operation_name(self, populated_store):
        spans = export_scan_traces_to_file(populated_store, "scan_otel_001")
        llm_spans = [s for s in spans if "llm.generate" in s["operationName"]]
        assert len(llm_spans) == 1
        assert llm_spans[0]["operationName"] == "llm.generate.step_1"

    def test_tool_span_operation_name(self, populated_store):
        spans = export_scan_traces_to_file(populated_store, "scan_otel_001")
        tool_names = {s["operationName"] for s in spans if "tool.execute" in s["operationName"]}
        assert "tool.execute.get_order" in tool_names
        assert "tool.execute.request_refund" in tool_names

    def test_spans_share_trace_id(self, populated_store):
        spans = export_scan_traces_to_file(populated_store, "scan_otel_001")
        trace_ids = {s["traceId"] for s in spans}
        # All spans in a scan share the same derived trace ID
        assert len(trace_ids) == 1

    def test_trace_id_is_deterministic(self, populated_store):
        spans1 = export_scan_traces_to_file(populated_store, "scan_otel_001")
        spans2 = export_scan_traces_to_file(populated_store, "scan_otel_001")
        assert spans1[0]["traceId"] == spans2[0]["traceId"]

    def test_llm_span_has_token_usage_attributes(self, populated_store):
        spans = export_scan_traces_to_file(populated_store, "scan_otel_001")
        llm_span = next(s for s in spans if "llm.generate" in s["operationName"])
        assert llm_span["attributes"]["llm.total_tokens"] == 150
        assert llm_span["attributes"]["llm.provider"] == "groq"

    def test_tool_span_denied_has_error_status(self, populated_store):
        spans = export_scan_traces_to_file(populated_store, "scan_otel_001")
        refund_span = next(s for s in spans if "request_refund" in s["operationName"])
        assert refund_span["status"]["code"] == "ERROR"

    def test_tool_span_allowed_has_ok_status(self, populated_store):
        spans = export_scan_traces_to_file(populated_store, "scan_otel_001")
        order_span = next(s for s in spans if "get_order" in s["operationName"])
        assert order_span["status"]["code"] == "OK"

    def test_export_to_file_writes_jsonl(self, populated_store, tmp_path):
        out = tmp_path / "traces.jsonl"
        spans = export_scan_traces_to_file(populated_store, "scan_otel_001", output_file=out)
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) == len(spans)
        # Each line must be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "traceId" in parsed

    def test_empty_scan_returns_no_spans(self):
        with TraceStore(":memory:") as store:
            spans = export_scan_traces_to_file(store, "scan_nonexistent")
        assert spans == []
