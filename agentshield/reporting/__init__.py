"""Security reporting (HTML, JSON, SARIF, CLI, regression) and comparative benchmarking."""

from agentshield.reporting.benchmark import (
    BenchmarkComparison,
    BenchmarkRunner,
    VariantBenchmarkSummary,
)
from agentshield.reporting.cli_report import render_cli_report
from agentshield.reporting.html_report import render_html_report, write_html_report
from agentshield.reporting.json_report import render_json_report, write_json_report
from agentshield.reporting.models import (
    REMEDIATION_BY_CATEGORY,
    AuditReportData,
    Confidence,
    FindingReportItem,
    build_audit_report,
)
from agentshield.reporting.regression import (
    RegressionRecord,
    RegressionStore,
    create_regressions_from_findings,
    create_regressions_from_scan_report,
    record_replay_result,
    replay_regression_record,
    replay_scan_regressions,
)
from agentshield.reporting.sarif_report import build_sarif, write_sarif_report

__all__ = [
    "AuditReportData",
    "BenchmarkComparison",
    "BenchmarkRunner",
    "Confidence",
    "FindingReportItem",
    "REMEDIATION_BY_CATEGORY",
    "RegressionRecord",
    "RegressionStore",
    "VariantBenchmarkSummary",
    "build_audit_report",
    "build_sarif",
    "create_regressions_from_findings",
    "create_regressions_from_scan_report",
    "record_replay_result",
    "replay_regression_record",
    "replay_scan_regressions",
    "render_cli_report",
    "render_html_report",
    "render_json_report",
    "write_html_report",
    "write_json_report",
    "write_sarif_report",
]
