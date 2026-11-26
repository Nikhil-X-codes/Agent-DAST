"""Regression replay — persist confirmed findings as versioned test cases and replay them.

Phase 8 add-on: Convert confirmed findings from a scan into versioned regression records
that can be replayed after a reviewed fix to verify remediation.

Design constraints (agent1.md Phase 8):
- Never auto-apply or deploy changes.
- Retain unresolved/inconclusive results.
- Replay must reproduce the original test case against the current target.
- Before/after comparison is stored as a RegressionRecord.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentshield.evaluation.models import Verdict
from agentshield.reporting.models import FindingReportItem

RegressionStatus = Literal["open", "fixed", "regressed", "inconclusive"]

_REGRESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS regressions (
    regression_id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL,
    test_id TEXT NOT NULL,
    test_title TEXT NOT NULL,
    category TEXT NOT NULL,
    original_verdict TEXT NOT NULL,
    original_severity TEXT NOT NULL,
    original_summary TEXT NOT NULL,
    replay_verdict TEXT,
    replay_summary TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL,
    replayed_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_regressions_scan ON regressions(scan_id);
CREATE INDEX IF NOT EXISTS idx_regressions_status ON regressions(status);
"""


class RegressionRecord(BaseModel):
    """A versioned regression test derived from a confirmed/suspected finding."""

    regression_id: str = Field(default_factory=lambda: f"reg_{uuid.uuid4().hex[:10]}")
    scan_id: str
    test_id: str
    test_title: str
    category: str
    original_verdict: Verdict
    original_severity: str
    original_summary: str
    replay_verdict: Verdict | None = None
    replay_summary: str | None = None
    status: RegressionStatus = "open"
    created_at: str = Field(default_factory=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    replayed_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_status(self) -> RegressionStatus:
        """Determine regression status from original vs. replay verdicts."""
        if self.replay_verdict is None:
            return "open"
        if self.original_verdict in {"confirmed", "suspected"}:
            if self.replay_verdict == "no_violation_observed":
                return "fixed"
            if self.replay_verdict in {"confirmed", "suspected"}:
                return "regressed"
        return "inconclusive"


class RegressionStore:
    """SQLite-backed store for regression records."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_REGRESSION_SCHEMA)
        self.conn.commit()

    def save(self, record: RegressionRecord) -> None:
        """Persist or update a regression record."""
        with self._lock, self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO regressions
                (regression_id, scan_id, test_id, test_title, category,
                 original_verdict, original_severity, original_summary,
                 replay_verdict, replay_summary, status, created_at, replayed_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.regression_id,
                    record.scan_id,
                    record.test_id,
                    record.test_title,
                    record.category,
                    record.original_verdict,
                    record.original_severity,
                    record.original_summary,
                    record.replay_verdict,
                    record.replay_summary,
                    record.status,
                    record.created_at,
                    record.replayed_at,
                    json.dumps(record.metadata),
                ),
            )

    def load_by_scan(self, scan_id: str) -> list[RegressionRecord]:
        """Return all regression records for a scan."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM regressions WHERE scan_id = ? ORDER BY created_at ASC",
                (scan_id,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def load_open(self) -> list[RegressionRecord]:
        """Return all open (not yet replayed/fixed) regression records."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM regressions WHERE status = 'open' ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def load_all(self) -> list[RegressionRecord]:
        """Return all regression records."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM regressions ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RegressionRecord:
        return RegressionRecord(
            regression_id=row["regression_id"],
            scan_id=row["scan_id"],
            test_id=row["test_id"],
            test_title=row["test_title"],
            category=row["category"],
            original_verdict=row["original_verdict"],
            original_severity=row["original_severity"],
            original_summary=row["original_summary"],
            replay_verdict=row["replay_verdict"],
            replay_summary=row["replay_summary"],
            status=row["status"],
            created_at=row["created_at"],
            replayed_at=row["replayed_at"],
            metadata=json.loads(row["metadata_json"]),
        )

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> RegressionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_regressions_from_findings(
    scan_id: str,
    findings: list[FindingReportItem],
    *,
    include_suspected: bool = True,
) -> list[RegressionRecord]:
    """Convert confirmed (and optionally suspected) findings into RegressionRecords.

    Only confirmed and suspected findings become regression cases.
    Inconclusive and no_violation_observed findings are skipped.
    """
    records: list[RegressionRecord] = []
    for f in findings:
        if f.verdict == "confirmed" or (include_suspected and f.verdict == "suspected"):
            records.append(
                RegressionRecord(
                    scan_id=scan_id,
                    test_id=f.test_id,
                    test_title=f.test_title,
                    category=f.category,
                    original_verdict=f.verdict,
                    original_severity=f.severity,
                    original_summary=f.summary,
                    metadata={
                        "evidence": f.evidence,
                        "tool_call_ids": f.tool_call_ids,
                        "evaluator": f.evaluator,
                        "limitations": f.limitations,
                    },
                )
            )
    return records


def create_regressions_from_scan_report(
    scan_report: Any,
    *,
    include_suspected: bool = True,
) -> list[RegressionRecord]:
    """Convert confirmed/suspected findings from a ScanReport, preserving prompts/turns for replay."""
    records: list[RegressionRecord] = []
    for ex in scan_report.executions:
        tc = ex.test_case
        for f in ex.report.findings:
            if f.verdict == "confirmed" or (include_suspected and f.verdict == "suspected"):
                records.append(
                    RegressionRecord(
                        scan_id=scan_report.scan_id,
                        test_id=tc.test_id,
                        test_title=tc.title,
                        category=f.category,
                        original_verdict=f.verdict,
                        original_severity=f.severity,
                        original_summary=f.summary,
                        metadata={
                            "prompt": tc.prompt,
                            "turns": tc.turns,
                            "expected_invariant": tc.expected_invariant,
                            "target_tools": tc.target_tools,
                            "evidence": f.evidence,
                            "tool_call_ids": f.tool_call_ids,
                            "evaluator": f.evaluator,
                            "limitations": f.limitations,
                        },
                    )
                )
    return records


def record_replay_result(
    record: RegressionRecord,
    *,
    replay_verdict: Verdict,
    replay_summary: str,
) -> RegressionRecord:
    """Update a regression record with the replay result and compute status."""
    import datetime

    replayed_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated = record.model_copy(
        update={
            "replay_verdict": replay_verdict,
            "replay_summary": replay_summary,
            "replayed_at": replayed_at,
        }
    )
    updated = updated.model_copy(update={"status": updated.compute_status()})
    return updated


def replay_regression_record(
    record: RegressionRecord,
    harness: Any,
    policy: Any,
    *,
    trace_store: Any | None = None,
    evaluator: Any | None = None,
) -> RegressionRecord:
    """Replay a saved RegressionRecord against a BaseHarness and update its verdict/status."""
    from agentshield.evaluation.evaluator import SecurityEvaluator
    from agentshield.evaluation.models import SecurityCategory
    from agentshield.test_engine.templates import build_static_templates
    from agentshield.tracing.collector import TraceCollector
    from agentshield.tracing.models import ScanSessionRecord
    from agentshield.tracing.store import TraceStore
    from agentshield.harnesses.base import SessionContext

    caps = harness.discover_capabilities()
    prompts: list[str] = []
    if record.metadata.get("turns"):
        prompts = [str(p) for p in record.metadata["turns"]]
    elif record.metadata.get("prompt"):
        prompts = [str(record.metadata["prompt"])]
    else:
        templates = build_static_templates(
            record.category,  # type: ignore[arg-type]
            tools=caps.tools,
            policy=policy,
        )
        if templates:
            prompts = list(templates[0].turns or [templates[0].prompt])
        else:
            prompts = [f"Replay security regression {record.test_id}: {record.test_title}"]

    owns_store = trace_store is None
    active_store = trace_store or TraceStore(":memory:")
    active_eval = evaluator or SecurityEvaluator()
    replay_scan_id = f"replay_{record.regression_id}"
    session = SessionContext(authenticated_subject=policy.authenticated_subject)

    try:
        active_store.ensure_session(
            ScanSessionRecord(
                scan_id=replay_scan_id,
                target_name=caps.target_name,
                target_type=caps.target_type,
                observability=caps.observability,
                telemetry_available=caps.telemetry_available,
                metadata={"replay_of": record.scan_id, "regression_id": record.regression_id},
            )
        )
        collector = TraceCollector(active_store)
        for p in prompts:
            collector.trace_execution(harness, p, scan_id=replay_scan_id, session=session)

        report = active_eval.evaluate_trace(
            active_store,
            replay_scan_id,
            policy,
            session_id=session.session_id,
            use_llm_judge=False,
        )
        matching = [f for f in report.findings if f.category == record.category]
        if matching:
            replay_verdict = matching[0].verdict
            replay_summary = matching[0].summary
        else:
            replay_verdict = report.overall_verdict
            replay_summary = f"Replay overall verdict: {report.overall_verdict}"
        return record_replay_result(
            record,
            replay_verdict=replay_verdict,
            replay_summary=replay_summary,
        )
    finally:
        if owns_store:
            active_store.close()


def replay_scan_regressions(
    store: RegressionStore,
    scan_id: str,
    harness: Any,
    policy: Any,
    *,
    trace_store: Any | None = None,
) -> list[RegressionRecord]:
    """Replay all regression records for a scan, persist updated statuses, and return them."""
    records = store.load_by_scan(scan_id)
    updated_records: list[RegressionRecord] = []
    for rec in records:
        updated = replay_regression_record(rec, harness, policy, trace_store=trace_store)
        store.save(updated)
        updated_records.append(updated)
    return updated_records

