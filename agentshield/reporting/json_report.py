"""JSON security audit report serializer."""

from __future__ import annotations

from pathlib import Path

from agentshield.reporting.models import AuditReportData


def render_json_report(report: AuditReportData) -> str:
    """Serialize AuditReportData to formatted JSON."""
    return report.model_dump_json(indent=2)


def write_json_report(report: AuditReportData, output_path: str | Path) -> Path:
    """Write a JSON audit report to disk and return the resolved path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_json_report(report), encoding="utf-8")
    return path
