"""SQLite persistence for AgentShield API scan jobs and generated reports."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from agentshield.api.schemas import ScanStatus, ScanStatusResponse
from agentshield.reporting.models import AuditReportData

_SCAN_JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_scans (
    scan_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    target_name TEXT NOT NULL,
    target_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    overall_verdict TEXT,
    tests_planned INTEGER NOT NULL DEFAULT 0,
    tests_executed INTEGER NOT NULL DEFAULT 0,
    total_tokens_used INTEGER NOT NULL DEFAULT 0,
    termination_reason TEXT,
    error TEXT,
    report_json TEXT,
    report_html TEXT
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ScanStore:
    """Thread-safe SQLite store for scan job lifecycle and generated reports."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCAN_JOB_SCHEMA)
        self.conn.commit()

    def create_scan(self, *, scan_id: str, target_name: str, target_type: str) -> ScanStatusResponse:
        now = _now_iso()
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO api_scans (scan_id, status, target_name, target_type, created_at, updated_at) "
                "VALUES (?, 'queued', ?, ?, ?, ?)",
                (scan_id, target_name, target_type, now, now),
            )
        return ScanStatusResponse(
            scan_id=scan_id,
            status="queued",
            target_name=target_name,
            target_type=target_type,
            created_at=now,
            updated_at=now,
        )

    def update_status(self, scan_id: str, status: ScanStatus, *, error: str | None = None) -> None:
        now = _now_iso()
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE api_scans SET status = ?, updated_at = ?, error = COALESCE(?, error) WHERE scan_id = ?",
                (status, now, error, scan_id),
            )

    def save_report(
        self,
        scan_id: str,
        *,
        audit_report: AuditReportData,
        html_report: str,
    ) -> None:
        now = _now_iso()
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE api_scans SET status = 'completed', updated_at = ?, overall_verdict = ?, "
                "tests_planned = ?, tests_executed = ?, total_tokens_used = ?, "
                "termination_reason = ?, report_json = ?, report_html = ? WHERE scan_id = ?",
                (
                    now,
                    audit_report.overall_verdict,
                    audit_report.total_tests,
                    audit_report.completed_tests,
                    audit_report.total_tokens_used,
                    audit_report.termination_reason,
                    audit_report.model_dump_json(),
                    html_report,
                    scan_id,
                ),
            )

    def get_status(self, scan_id: str) -> ScanStatusResponse | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT scan_id, status, target_name, target_type, created_at, updated_at, "
                "overall_verdict, tests_planned, tests_executed, total_tokens_used, termination_reason, error "
                "FROM api_scans WHERE scan_id = ?",
                (scan_id,),
            ).fetchone()
        if row is None:
            return None
        return ScanStatusResponse(**dict(row))

    def get_report(self, scan_id: str) -> tuple[AuditReportData | None, str | None]:
        with self._lock:
            row = self.conn.execute(
                "SELECT report_json, report_html FROM api_scans WHERE scan_id = ?",
                (scan_id,),
            ).fetchone()
        if row is None or not row["report_json"]:
            return None, None
        return AuditReportData.model_validate_json(row["report_json"]), row["report_html"]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
