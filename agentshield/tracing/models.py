"""Pydantic models for runtime execution tracing."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentshield.providers.base import TokenUsage

AuthorizationDecision = Literal["allowed", "denied", "missing", "unavailable"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ScanSessionRecord(BaseModel):
    """Metadata for an audited scan session stored in SQLite."""

    scan_id: str
    target_name: str
    target_type: str
    observability: str
    telemetry_available: bool
    created_at: str = Field(default_factory=_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMTraceEvent(BaseModel):
    """Recorded LLM inference step correlated to a scan and session."""

    event_id: str = Field(default_factory=lambda: f"llm_{uuid.uuid4().hex[:12]}")
    scan_id: str
    session_id: str
    step: int = 1
    provider: str | None = None
    model: str | None = None
    prompt_input: Any = None
    response_output: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    usage: TokenUsage | None = None
    telemetry_available: bool = True
    error: str | None = None
    created_at: str = Field(default_factory=_now_iso)


class ToolTraceEvent(BaseModel):
    """Recorded tool execution event with authorization and DB state-change evidence."""

    event_id: str = Field(default_factory=lambda: f"tool_{uuid.uuid4().hex[:12]}")
    scan_id: str
    session_id: str
    tool_call_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    authorization: AuthorizationDecision = "unavailable"
    execution_status: str = "completed"
    state_changed: bool | None = None
    latency_ms: float | None = None
    error: str | None = None
    created_at: str = Field(default_factory=_now_iso)
