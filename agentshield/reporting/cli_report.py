"""Rich CLI renderer for security audit reports and benchmarks."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentshield.reporting.models import AuditReportData


def render_cli_report(report: AuditReportData, *, console: Console | None = None) -> None:
    """Print a structured security audit summary to the terminal via Rich."""
    out = console or Console()
    out.print(
        Panel(
            f"Target: [bold]{report.target_name}[/bold] ({report.target_type} / {report.observability})\n"
            f"Tests: {report.completed_tests}/{report.total_tests} completed ({report.incomplete_tests} incomplete)\n"
            f"Confirmed: [red]{report.confirmed_count}[/red] | Suspected: [yellow]{report.suspected_count}[/yellow] | "
            f"Inconclusive: {report.inconclusive_count} | No violation: [green]{report.no_violation_count}[/green]\n"
            f"Tokens: {report.total_tokens_used} | Avg Latency: {report.avg_latency_ms} ms | "
            f"API Cost: {report.api_expenditure_display}",
            title=f"AgentShield Audit Report — {report.scan_id} ({report.overall_verdict})",
        )
    )

    if report.findings:
        table = Table(title="Findings & Remediation")
        table.add_column("Category")
        table.add_column("Verdict")
        table.add_column("Severity")
        table.add_column("Confidence")
        table.add_column("OWASP / MITRE")
        table.add_column("Summary")
        for item in report.findings:
            table.add_row(
                item.category,
                item.verdict,
                item.severity,
                item.confidence,
                f"{item.taxonomy.owasp_llm}\n{item.taxonomy.mitre_atlas}",
                item.summary,
            )
        out.print(table)
