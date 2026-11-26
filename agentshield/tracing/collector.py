"""Runtime trace collector integrating harnesses, SQLite TraceStore, and optional OpenTelemetry."""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from typing import Any

from agentshield.harnesses.base import (
    BaseHarness,
    SessionContext,
    StandardizedExecutionResult,
    ToolCallTrace,
)
from agentshield.providers.base import TokenUsage
from agentshield.tracing.models import (
    AuthorizationDecision,
    LLMTraceEvent,
    ScanSessionRecord,
    ToolTraceEvent,
)
from agentshield.tracing.store import TraceStore

try:
    from opentelemetry import trace as _otel_trace
except ImportError:  # pragma: no cover
    _otel_trace = None


def _resolve_authorization(trace: ToolCallTrace, *, variant: str | None) -> AuthorizationDecision:
    if trace.authorized is False:
        return "denied"
    if trace.authorized is True:
        if variant == "vulnerable":
            return "missing"
        return "allowed"
    return "unavailable"


class TraceCollector:
    """Collects and persists structured runtime execution traces for any BaseHarness target."""

    def __init__(self, store: TraceStore, *, enable_otel: bool = True) -> None:
        self.store = store
        self._tracer = _otel_trace.get_tracer("agentshield") if (enable_otel and _otel_trace is not None) else None

    def trace_execution(
        self,
        harness: BaseHarness,
        prompt: str,
        *,
        scan_id: str | None = None,
        session: SessionContext | None = None,
    ) -> StandardizedExecutionResult:
        """Execute a prompt through a harness and persist correlated LLM & tool traces."""
        active_scan_id = scan_id or f"scan_{uuid.uuid4().hex[:8]}"
        active_session = session or SessionContext()

        caps = harness.discover_capabilities()
        self.store.ensure_session(
            ScanSessionRecord(
                scan_id=active_scan_id,
                target_name=caps.target_name,
                target_type=caps.target_type,
                observability=caps.observability,
                telemetry_available=caps.telemetry_available,
                metadata={"session_handling": caps.session_handling},
            )
        )

        span_ctx: Any = (
            self._tracer.start_as_current_span(
                "agentshield.harness.execute",
                attributes={
                    "agentshield.scan_id": active_scan_id,
                    "agentshield.session_id": active_session.session_id,
                    "agentshield.target_name": caps.target_name,
                    "agentshield.target_type": caps.target_type,
                },
            )
            if self._tracer is not None
            else nullcontext()
        )

        with span_ctx:
            try:
                result = harness.execute(prompt, session=active_session)
            except Exception as exc:
                self.store.record_llm_event(
                    LLMTraceEvent(
                        scan_id=active_scan_id,
                        session_id=active_session.session_id,
                        step=1,
                        prompt_input=prompt,
                        telemetry_available=caps.telemetry_available,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                raise

        self._persist_result_events(active_scan_id, result)
        return result

    def _persist_result_events(self, scan_id: str, result: StandardizedExecutionResult) -> None:
        llm_turns = result.raw_metadata.get("llm_turns")
        if isinstance(llm_turns, list) and llm_turns:
            for turn in llm_turns:
                if not isinstance(turn, dict):
                    continue
                raw_usage = turn.get("usage")
                usage = TokenUsage(**raw_usage) if isinstance(raw_usage, dict) else None
                self.store.record_llm_event(
                    LLMTraceEvent(
                        scan_id=scan_id,
                        session_id=result.session.session_id,
                        step=int(turn.get("step", 1)),
                        provider=turn.get("provider") or result.provider,
                        model=turn.get("model") or result.model,
                        prompt_input=turn.get("input_messages"),
                        response_output=turn.get("response_output"),
                        tool_calls=list(turn.get("tool_calls") or []),
                        latency_ms=float(turn.get("latency_ms", 0.0)),
                        usage=usage,
                        telemetry_available=result.telemetry_available,
                        error=result.error,
                    )
                )
        else:
            self.store.record_llm_event(
                LLMTraceEvent(
                    scan_id=scan_id,
                    session_id=result.session.session_id,
                    step=1,
                    provider=result.provider,
                    model=result.model,
                    prompt_input=result.prompt,
                    response_output=result.response_text,
                    latency_ms=result.latency_ms,
                    usage=result.usage,
                    telemetry_available=result.telemetry_available,
                    error=result.error,
                )
            )

        if result.execution_trace is None:
            return

        variant = result.raw_metadata.get("variant")
        for trace in result.execution_trace:
            status_map = "completed" if trace.status == "success" else (trace.status or "unknown")
            self.store.record_tool_event(
                ToolTraceEvent(
                    scan_id=scan_id,
                    session_id=result.session.session_id,
                    tool_call_id=trace.tool_call_id,
                    tool_name=trace.tool_name,
                    arguments=trace.arguments,
                    result=trace.output,
                    authorization=_resolve_authorization(trace, variant=variant if isinstance(variant, str) else None),
                    execution_status=status_map,
                    state_changed=trace.state_changed,
                    latency_ms=trace.latency_ms,
                )
            )
