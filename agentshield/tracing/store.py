"""SQLite persistence store for AgentShield runtime execution traces."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from agentshield.providers.base import TokenUsage
from agentshield.tracing.models import LLMTraceEvent, ScanSessionRecord, ToolTraceEvent
from agentshield.tracing.redactor import redact_sensitive

_TRACE_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_sessions (
    scan_id TEXT PRIMARY KEY,
    target_name TEXT NOT NULL,
    target_type TEXT NOT NULL,
    observability TEXT NOT NULL,
    telemetry_available INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_events (
    event_id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    provider TEXT,
    model TEXT,
    prompt_input_json TEXT,
    response_output TEXT,
    tool_calls_json TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    usage_json TEXT,
    telemetry_available INTEGER NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (scan_id) REFERENCES scan_sessions(scan_id)
);

CREATE TABLE IF NOT EXISTS tool_events (
    event_id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    tool_call_id TEXT,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    result_json TEXT,
    authorization TEXT NOT NULL,
    execution_status TEXT NOT NULL,
    state_changed INTEGER,
    latency_ms REAL,
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (scan_id) REFERENCES scan_sessions(scan_id)
);

CREATE INDEX IF NOT EXISTS idx_llm_scan_session ON llm_events (scan_id, session_id);
CREATE INDEX IF NOT EXISTS idx_tool_scan_session ON tool_events (scan_id, session_id);
CREATE INDEX IF NOT EXISTS idx_tool_call_id ON tool_events (tool_call_id);
"""


class TraceStore:
    """Persists and queries redacted scan sessions, LLM events, and tool events in SQLite."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.executescript(_TRACE_SCHEMA)
        self.conn.commit()

    def ensure_session(self, session: ScanSessionRecord) -> None:
        safe_meta = redact_sensitive(session.metadata)
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO scan_sessions "
                "(scan_id, target_name, target_type, observability, telemetry_available, created_at, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session.scan_id,
                    session.target_name,
                    session.target_type,
                    session.observability,
                    1 if session.telemetry_available else 0,
                    session.created_at,
                    json.dumps(safe_meta),
                ),
            )

    def record_llm_event(self, event: LLMTraceEvent) -> None:
        safe_prompt = redact_sensitive(event.prompt_input)
        safe_output = redact_sensitive(event.response_output)
        safe_tool_calls = redact_sensitive(event.tool_calls)
        safe_error = redact_sensitive(event.error)
        usage_json = json.dumps(event.usage.model_dump()) if event.usage is not None else None

        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO llm_events "
                "(event_id, scan_id, session_id, step, provider, model, prompt_input_json, "
                "response_output, tool_calls_json, latency_ms, usage_json, telemetry_available, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.scan_id,
                    event.session_id,
                    event.step,
                    event.provider,
                    event.model,
                    json.dumps(safe_prompt),
                    safe_output,
                    json.dumps(safe_tool_calls),
                    event.latency_ms,
                    usage_json,
                    1 if event.telemetry_available else 0,
                    safe_error,
                    event.created_at,
                ),
            )

    def record_tool_event(self, event: ToolTraceEvent) -> None:
        safe_args = redact_sensitive(event.arguments)
        safe_result = redact_sensitive(event.result)
        safe_error = redact_sensitive(event.error)
        state_int = None if event.state_changed is None else (1 if event.state_changed else 0)

        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO tool_events "
                "(event_id, scan_id, session_id, tool_call_id, tool_name, arguments_json, result_json, "
                "authorization, execution_status, state_changed, latency_ms, error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.scan_id,
                    event.session_id,
                    event.tool_call_id,
                    event.tool_name,
                    json.dumps(safe_args),
                    json.dumps(safe_result) if safe_result is not None else None,
                    event.authorization,
                    event.execution_status,
                    state_int,
                    event.latency_ms,
                    safe_error,
                    event.created_at,
                ),
            )

    def get_session(self, scan_id: str) -> ScanSessionRecord | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM scan_sessions WHERE scan_id = ?", (scan_id,)).fetchone()
        if row is None:
            return None
        return ScanSessionRecord(
            scan_id=row["scan_id"],
            target_name=row["target_name"],
            target_type=row["target_type"],
            observability=row["observability"],
            telemetry_available=bool(row["telemetry_available"]),
            created_at=row["created_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    def get_llm_events(self, scan_id: str, *, session_id: str | None = None) -> list[LLMTraceEvent]:
        query = "SELECT * FROM llm_events WHERE scan_id = ?"
        params: list[Any] = [scan_id]
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY step ASC, created_at ASC"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [
            LLMTraceEvent(
                event_id=r["event_id"],
                scan_id=r["scan_id"],
                session_id=r["session_id"],
                step=r["step"],
                provider=r["provider"],
                model=r["model"],
                prompt_input=json.loads(r["prompt_input_json"]) if r["prompt_input_json"] else None,
                response_output=r["response_output"],
                tool_calls=json.loads(r["tool_calls_json"]),
                latency_ms=r["latency_ms"],
                usage=TokenUsage(**json.loads(r["usage_json"])) if r["usage_json"] else None,
                telemetry_available=bool(r["telemetry_available"]),
                error=r["error"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_tool_events(self, scan_id: str, *, session_id: str | None = None) -> list[ToolTraceEvent]:
        query = "SELECT * FROM tool_events WHERE scan_id = ?"
        params: list[Any] = [scan_id]
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at ASC"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [
            ToolTraceEvent(
                event_id=r["event_id"],
                scan_id=r["scan_id"],
                session_id=r["session_id"],
                tool_call_id=r["tool_call_id"],
                tool_name=r["tool_name"],
                arguments=json.loads(r["arguments_json"]),
                result=json.loads(r["result_json"]) if r["result_json"] is not None else None,
                authorization=r["authorization"],
                execution_status=r["execution_status"],
                state_changed=None if r["state_changed"] is None else bool(r["state_changed"]),
                latency_ms=r["latency_ms"],
                error=r["error"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
