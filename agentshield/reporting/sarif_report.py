"""SARIF 2.1.0 export for AgentShield security scan findings.

Generates Static Analysis Results Interchange Format (SARIF) output from AuditReportData.
Compatible with GitHub Advanced Security, VS Code SARIF Viewer, and other CI integrations.

Design constraints (agent1.md Phase 6):
- Stable rule IDs per security category (never change between versions).
- Source locations are OMITTED when not known (not fabricated).
- Each rule links to its OWASP LLM reference.
- Output is a single SARIF 2.1.0 JSON document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentshield.evaluation.models import SecurityCategory
from agentshield.reporting.models import AuditReportData, FindingReportItem

# Stable rule IDs — these MUST NOT change between versions.
# Format: agentshield/<category>
_RULE_IDS: dict[SecurityCategory, str] = {
    "unauthorized_tool_execution": "agentshield/unauthorized-tool-execution",
    "sensitive_information_disclosure": "agentshield/sensitive-information-disclosure",
    "excessive_agency": "agentshield/excessive-agency",
    "data_isolation_failure": "agentshield/data-isolation-failure",
    "resource_consumption": "agentshield/resource-consumption",
    "indirect_prompt_injection": "agentshield/indirect-prompt-injection",
}

_RULE_NAMES: dict[SecurityCategory, str] = {
    "unauthorized_tool_execution": "UnauthorizedToolExecution",
    "sensitive_information_disclosure": "SensitiveInformationDisclosure",
    "excessive_agency": "ExcessiveAgency",
    "data_isolation_failure": "DataIsolationFailure",
    "resource_consumption": "ResourceConsumption",
    "indirect_prompt_injection": "IndirectPromptInjection",
}

# Map verdict to SARIF result kind and level
_VERDICT_TO_KIND: dict[str, str] = {
    "confirmed": "fail",
    "suspected": "open",
    "inconclusive": "open",
    "no_violation_observed": "pass",
}

_SEVERITY_TO_LEVEL: dict[str, str] = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "note",
    "info": "note",
}

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
SARIF_VERSION = "2.1.0"
TOOL_NAME = "AgentShield"
TOOL_VERSION = "0.1.0"
TOOL_URI = "https://github.com/Nikhil-X-codes/Agent-DAST"


def _build_rule(category: SecurityCategory, finding: FindingReportItem) -> dict[str, Any]:
    """Build a SARIF reportingDescriptor for a security category."""
    rule_id = _RULE_IDS[category]
    return {
        "id": rule_id,
        "name": _RULE_NAMES[category],
        "shortDescription": {"text": finding.taxonomy.owasp_llm},
        "fullDescription": {
            "text": (
                f"{finding.taxonomy.owasp_llm} | {finding.taxonomy.mitre_atlas}. "
                f"Remediation: {finding.remediation}"
            )
        },
        "help": {
            "text": finding.remediation,
            "markdown": f"**Remediation:** {finding.remediation}",
        },
        "properties": {
            "owasp_llm": finding.taxonomy.owasp_llm,
            "mitre_atlas": finding.taxonomy.mitre_atlas,
            "tags": [category, "security", "ai-agent"],
        },
    }


def _build_result(finding: FindingReportItem, scan_id: str) -> dict[str, Any]:
    """Build a SARIF result object from a FindingReportItem."""
    rule_id = _RULE_IDS[finding.category]
    kind = _VERDICT_TO_KIND.get(finding.verdict, "open")
    level = _SEVERITY_TO_LEVEL.get(finding.severity, "note")

    # Build message text with evidence
    evidence_text = ""
    if finding.evidence:
        evidence_text = "\nEvidence:\n" + "\n".join(f"  - {e}" for e in finding.evidence)
    if finding.limitations:
        evidence_text += "\nLimitations:\n" + "\n".join(f"  - {l}" for l in finding.limitations)

    message_text = (
        f"[{finding.verdict.upper()}] {finding.summary}"
        f"\nScan: {scan_id} | Test: {finding.test_id} | Confidence: {finding.confidence}"
        f"{evidence_text}"
    )

    result: dict[str, Any] = {
        "ruleId": rule_id,
        "kind": kind,
        "level": level,
        "message": {"text": message_text},
        "properties": {
            "verdict": finding.verdict,
            "severity": finding.severity,
            "confidence": finding.confidence,
            "test_id": finding.test_id,
            "test_title": finding.test_title,
            "evaluator": finding.evaluator,
            "tool_call_ids": finding.tool_call_ids,
        },
    }

    # NOTE: source locations are intentionally omitted — we do not know file/line
    # and must not fabricate them (agent1.md Phase 6).

    return result


def build_sarif(report: AuditReportData) -> dict[str, Any]:
    """Build a SARIF 2.1.0 document from an AuditReportData instance.

    Returns the SARIF document as a Python dict (serialize with json.dumps).
    Rules are deduplicated by category; source locations are omitted.
    """
    # Collect unique rules (one per category seen in findings)
    seen_categories: set[str] = set()
    rules: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for finding in report.findings:
        cat = finding.category
        if cat not in seen_categories:
            seen_categories.add(cat)
            rules.append(_build_rule(cat, finding))
        results.append(_build_result(finding, report.scan_id))

    # If no findings, still emit all rules so CI sees the scan ran
    if not seen_categories:
        # Build placeholder rules from categories with no findings
        from agentshield.evaluation.models import TAXONOMY_BY_CATEGORY, SecurityCategory
        from agentshield.reporting.models import REMEDIATION_BY_CATEGORY

        for cat, taxonomy in TAXONOMY_BY_CATEGORY.items():
            rules.append({
                "id": _RULE_IDS[cat],
                "name": _RULE_NAMES[cat],
                "shortDescription": {"text": taxonomy.owasp_llm},
                "fullDescription": {"text": f"{taxonomy.owasp_llm} | {taxonomy.mitre_atlas}"},
                "help": {"text": REMEDIATION_BY_CATEGORY[cat]},
                "properties": {
                    "owasp_llm": taxonomy.owasp_llm,
                    "mitre_atlas": taxonomy.mitre_atlas,
                    "tags": [cat, "security", "ai-agent"],
                },
            })

    tool_component: dict[str, Any] = {
        "name": TOOL_NAME,
        "version": TOOL_VERSION,
        "informationUri": TOOL_URI,
        "rules": rules,
    }

    run: dict[str, Any] = {
        "tool": {"driver": tool_component},
        "results": results,
        "properties": {
            "scan_id": report.scan_id,
            "target_name": report.target_name,
            "target_type": report.target_type,
            "observability": report.observability,
            "telemetry_available": report.telemetry_available,
            "overall_verdict": report.overall_verdict,
            "total_tests": report.total_tests,
            "completed_tests": report.completed_tests,
            "confirmed_count": report.confirmed_count,
            "total_tokens_used": report.total_tokens_used,
            "started_at": report.started_at,
            "completed_at": report.completed_at,
            "api_expenditure": report.api_expenditure_display,
        },
    }

    return {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [run],
    }


def write_sarif_report(report: AuditReportData, output_path: str | Path) -> Path:
    """Write a SARIF 2.1.0 JSON report to disk and return the resolved path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sarif_doc = build_sarif(report)
    path.write_text(json.dumps(sarif_doc, indent=2), encoding="utf-8")
    return path
