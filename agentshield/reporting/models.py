"""Structured security audit report models built strictly from measured scan evidence."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentshield.evaluation.models import SecurityCategory, Severity, TaxonomyMapping, Verdict
from agentshield.orchestrator.models import ScanReport
from agentshield.tracing.store import TraceStore

Confidence = Literal["high", "medium", "low"]

REMEDIATION_BY_CATEGORY: dict[SecurityCategory, str] = {
    "unauthorized_tool_execution": (
        "Enforce server-side authorization and session ownership checks outside the LLM "
        "in the tool execution layer before mutating state."
    ),
    "sensitive_information_disclosure": (
        "Remove sensitive secrets/canaries from LLM-accessible context, apply output redaction, "
        "and restrict read tools to authorized tenant records."
    ),
    "excessive_agency": (
        "Enforce strict least-privilege tool allowlists and schema/argument validation "
        "at the runtime boundary."
    ),
    "data_isolation_failure": (
        "Bind authenticated_subject server-side and scope all database queries and mutations "
        "with mandatory tenant ownership predicates."
    ),
    "resource_consumption": (
        "Configure hard per-scan token, turn, and step budgets in ScanOrchestrator and "
        "enforce max_tokens on provider requests."
    ),
    "indirect_prompt_injection": (
        "Treat retrieved external/FAQ/document content as untrusted data and require "
        "server-side authorization for any downstream tool invocation."
    ),
}


class FindingReportItem(BaseModel):
    """Enriched finding entry with confidence, remediation, and trace evidence."""

    test_id: str
    test_title: str
    category: SecurityCategory
    verdict: Verdict
    severity: Severity
    confidence: Confidence
    summary: str
    taxonomy: TaxonomyMapping
    evaluator: str
    evidence: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    remediation: str


class AuditReportData(BaseModel):
    """Complete, strictly measured audit report ready for HTML, JSON, or CLI rendering."""

    scan_id: str
    target_name: str
    target_type: str
    observability: str
    telemetry_available: bool
    providers_used: list[str] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list)
    policies_tested: list[str] = Field(default_factory=list)
    started_at: str
    completed_at: str
    termination_reason: str
    overall_verdict: Verdict
    total_tests: int
    completed_tests: int
    incomplete_tests: int
    confirmed_count: int
    suspected_count: int
    inconclusive_count: int
    no_violation_count: int
    severity_counts: dict[str, int] = Field(default_factory=dict)
    total_tokens_used: int
    total_turns_executed: int
    total_latency_ms: float
    avg_latency_ms: float
    api_expenditure_display: str = "₹0 (Zero-Cost)"
    findings: list[FindingReportItem] = Field(default_factory=list)
    trace_highlights: list[dict[str, Any]] = Field(default_factory=list)


def _compute_confidence(*, verdict: Verdict, evaluator: str, telemetry_available: bool) -> Confidence:
    if verdict == "inconclusive" or not telemetry_available:
        return "low"
    if evaluator == "deterministic" and verdict in {"confirmed", "no_violation_observed"}:
        return "high"
    return "medium"


def get_confirmed_critical_or_high(report: AuditReportData) -> list[FindingReportItem]:
    """Return confirmed findings with critical or high severity (used by CI failure gates)."""
    return [
        f for f in report.findings
        if f.verdict == "confirmed" and f.severity in {"critical", "high"}
    ]


def has_confirmed_critical_or_high(report: AuditReportData) -> bool:
    """Return True if the audit report contains any confirmed critical or high finding."""
    return bool(get_confirmed_critical_or_high(report))


def build_audit_report(
    scan_report: ScanReport,
    *,
    trace_store: TraceStore | None = None,
) -> AuditReportData:
    """Transform a ScanReport (and optional TraceStore) into AuditReportData without fabrication."""
    providers: list[str] = []
    models: list[str] = []
    latencies: list[float] = []
    trace_highlights: list[dict[str, Any]] = []

    if trace_store is not None:
        llm_events = trace_store.get_llm_events(scan_report.scan_id)
        for ev in llm_events:
            if ev.provider and ev.provider not in providers:
                providers.append(ev.provider)
            if ev.model and ev.model not in models:
                models.append(ev.model)
            if ev.latency_ms > 0:
                latencies.append(ev.latency_ms)

        for tev in trace_store.get_tool_events(scan_report.scan_id):
            trace_highlights.append(
                {
                    "session_id": tev.session_id,
                    "tool_call_id": tev.tool_call_id,
                    "tool_name": tev.tool_name,
                    "authorization": tev.authorization,
                    "execution_status": tev.execution_status,
                    "state_changed": tev.state_changed,
                    "latency_ms": tev.latency_ms,
                }
            )

    findings: list[FindingReportItem] = []
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    confirmed_count = 0
    suspected_count = 0
    inconclusive_count = 0
    no_violation_count = 0

    for ex in scan_report.executions:
        if ex.report.overall_verdict == "confirmed":
            confirmed_count += 1
        elif ex.report.overall_verdict == "suspected":
            suspected_count += 1
        elif ex.report.overall_verdict == "inconclusive":
            inconclusive_count += 1
        else:
            no_violation_count += 1

        for f in ex.report.findings:
            if f.verdict in {"confirmed", "suspected"} and f.severity in severity_counts:
                severity_counts[f.severity] += 1
            if f.verdict == "no_violation_observed" and not f.evidence:
                continue
            findings.append(
                FindingReportItem(
                    test_id=ex.test_case.test_id,
                    test_title=ex.test_case.title,
                    category=f.category,
                    verdict=f.verdict,
                    severity=f.severity,
                    confidence=_compute_confidence(
                        verdict=f.verdict,
                        evaluator=f.evaluator,
                        telemetry_available=scan_report.telemetry_available,
                    ),
                    summary=f.summary,
                    taxonomy=f.taxonomy,
                    evaluator=f.evaluator,
                    evidence=f.evidence,
                    tool_call_ids=f.tool_call_ids,
                    limitations=f.limitations,
                    remediation=REMEDIATION_BY_CATEGORY[f.category],
                )
            )

    total_latency_ms = round(sum(latencies), 2)
    avg_latency_ms = round(total_latency_ms / len(latencies), 2) if latencies else 0.0
    incomplete_tests = max(scan_report.tests_planned - scan_report.tests_executed, 0)

    return AuditReportData(
        scan_id=scan_report.scan_id,
        target_name=scan_report.target_name,
        target_type=scan_report.target_type,
        observability=scan_report.observability,
        telemetry_available=scan_report.telemetry_available,
        providers_used=providers,
        models_used=models,
        policies_tested=[str(c) for c in scan_report.plan.selected_categories],
        started_at=scan_report.started_at,
        completed_at=scan_report.completed_at,
        termination_reason=scan_report.termination_reason,
        overall_verdict=scan_report.overall_verdict,
        total_tests=scan_report.tests_planned,
        completed_tests=scan_report.tests_executed,
        incomplete_tests=incomplete_tests,
        confirmed_count=confirmed_count,
        suspected_count=suspected_count,
        inconclusive_count=inconclusive_count,
        no_violation_count=no_violation_count,
        severity_counts=severity_counts,
        total_tokens_used=scan_report.total_tokens_used,
        total_turns_executed=scan_report.total_turns_executed,
        total_latency_ms=total_latency_ms,
        avg_latency_ms=avg_latency_ms,
        findings=findings,
        trace_highlights=trace_highlights,
    )
