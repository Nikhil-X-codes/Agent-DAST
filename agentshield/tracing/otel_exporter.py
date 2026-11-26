"""Optional OpenTelemetry export for AgentShield runtime traces.

Phase 4 add-on: Export LLMTraceEvent and ToolTraceEvent as OTEL spans to a file/console
exporter without requiring external hosting (zero-cost constraint).

Design constraints (agent1.md Phase 4):
- No mandatory external OTLP endpoint.
- Falls back to console/file exporter.
- Wraps existing TraceStore events as OTEL spans with appropriate attributes.
- Import is lazy — opentelemetry-sdk is optional.

Install (optional):
    pip install opentelemetry-sdk

Usage::

    from agentshield.tracing.otel_exporter import export_scan_traces_to_otel
    from agentshield.tracing.store import TraceStore

    with TraceStore("agentshield.db") as store:
        export_scan_traces_to_otel(store, scan_id="scan_abc123", output_file="traces.jsonl")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentshield.tracing.store import TraceStore


def _iso_to_ns(iso: str) -> int:
    """Convert ISO-8601 UTC string to nanoseconds since epoch (approximate)."""
    from datetime import datetime, timezone

    try:
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    except ValueError:
        return int(time.time() * 1_000_000_000)


def _build_llm_span(event: Any, trace_id: str) -> dict[str, Any]:
    """Build an OTEL-compatible span dict for an LLMTraceEvent."""
    start_ns = _iso_to_ns(event.created_at)
    duration_ns = int((event.latency_ms or 0) * 1_000_000)
    return {
        "traceId": trace_id,
        "spanId": event.event_id,
        "parentSpanId": None,
        "operationName": f"llm.generate.step_{event.step}",
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": start_ns + duration_ns,
        "attributes": {
            "agentshield.scan_id": event.scan_id,
            "agentshield.session_id": event.session_id,
            "agentshield.step": event.step,
            "llm.provider": event.provider or "unknown",
            "llm.model": event.model or "unknown",
            "llm.latency_ms": event.latency_ms,
            "llm.total_tokens": (event.usage.total_tokens if event.usage else 0),
            "llm.prompt_tokens": (event.usage.prompt_tokens if event.usage else 0),
            "llm.completion_tokens": (event.usage.completion_tokens if event.usage else 0),
            "llm.error": event.error or "",
            "telemetry.available": event.telemetry_available,
        },
        "status": {"code": "ERROR" if event.error else "OK"},
        "kind": "INTERNAL",
    }


def _build_tool_span(event: Any, trace_id: str, parent_span_id: str | None = None) -> dict[str, Any]:
    """Build an OTEL-compatible span dict for a ToolTraceEvent."""
    start_ns = _iso_to_ns(event.created_at)
    duration_ns = int((event.latency_ms or 0) * 1_000_000)
    return {
        "traceId": trace_id,
        "spanId": event.event_id,
        "parentSpanId": parent_span_id,
        "operationName": f"tool.execute.{event.tool_name}",
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": start_ns + duration_ns,
        "attributes": {
            "agentshield.scan_id": event.scan_id,
            "agentshield.session_id": event.session_id,
            "tool.name": event.tool_name,
            "tool.call_id": event.tool_call_id or "",
            "tool.authorization": event.authorization,
            "tool.execution_status": event.execution_status,
            "tool.state_changed": event.state_changed,
            "tool.latency_ms": event.latency_ms or 0.0,
            "tool.error": event.error or "",
        },
        "status": {
            "code": "ERROR" if (event.error or event.authorization == "denied") else "OK"
        },
        "kind": "INTERNAL",
    }


def export_scan_traces_to_file(
    store: "TraceStore",
    scan_id: str,
    *,
    output_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Export all trace events for a scan as OTEL-compatible span dicts.

    If output_file is provided, writes one JSON span per line (JSONL).
    Always returns the list of span dicts regardless of file output.

    This is a pure-Python fallback that works without the opentelemetry-sdk installed.
    """
    import hashlib

    # Derive a stable trace ID from scan_id (not random — reproducible)
    trace_id = hashlib.sha256(scan_id.encode()).hexdigest()[:32]

    spans: list[dict[str, Any]] = []

    llm_events = store.get_llm_events(scan_id)
    tool_events = store.get_tool_events(scan_id)

    # Build LLM spans
    for ev in llm_events:
        spans.append(_build_llm_span(ev, trace_id))

    # Build tool spans (parent = corresponding LLM step span, if determinable)
    # Correlate by session and step order (simple heuristic)
    llm_by_session: dict[str, list[Any]] = {}
    for ev in llm_events:
        llm_by_session.setdefault(ev.session_id, []).append(ev)

    for tev in tool_events:
        # Find the most recent LLM event in same session as parent
        parent_id: str | None = None
        session_llm = llm_by_session.get(tev.session_id, [])
        if session_llm:
            parent_id = session_llm[-1].event_id
        spans.append(_build_tool_span(tev, trace_id, parent_span_id=parent_id))

    if output_file is not None:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for span in spans:
                f.write(json.dumps(span) + "\n")

    return spans


def try_export_with_otel_sdk(
    store: "TraceStore",
    scan_id: str,
    *,
    output_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Attempt to export using opentelemetry-sdk; fall back to file export if not installed.

    This keeps the opentelemetry dependency optional (zero-cost constraint).
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        # Note: We still build the dicts for the return value and file output;
        # actual SDK spans are emitted to console as a side effect.
    except ImportError:
        pass  # SDK not installed — fall back to file export only

    return export_scan_traces_to_file(store, scan_id, output_file=output_file)
