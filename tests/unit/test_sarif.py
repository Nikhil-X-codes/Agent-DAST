"""Unit tests for SARIF 2.1.0 export (Phase 6 add-on)."""

from __future__ import annotations

import json

import pytest

from agentshield.evaluation.models import TAXONOMY_BY_CATEGORY
from agentshield.reporting.sarif_report import (
    SARIF_VERSION,
    build_sarif,
    write_sarif_report,
    _RULE_IDS,
)
from agentshield.reporting.models import AuditReportData, FindingReportItem, REMEDIATION_BY_CATEGORY


def _make_minimal_report(
    findings: list[FindingReportItem] | None = None,
    overall_verdict: str = "confirmed",
) -> AuditReportData:
    return AuditReportData(
        scan_id="scan_sarif_test",
        target_name="support-vulnerable",
        target_type="langgraph",
        observability="instrumented",
        telemetry_available=True,
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:01:00Z",
        termination_reason="all_tests_completed",
        overall_verdict=overall_verdict,  # type: ignore[arg-type]
        total_tests=3,
        completed_tests=3,
        incomplete_tests=0,
        confirmed_count=1,
        suspected_count=0,
        inconclusive_count=0,
        no_violation_count=2,
        total_tokens_used=1500,
        total_turns_executed=3,
        total_latency_ms=820.0,
        avg_latency_ms=273.3,
        findings=findings or [],
    )


def _make_finding(
    category: str = "unauthorized_tool_execution",
    verdict: str = "confirmed",
    severity: str = "critical",
) -> FindingReportItem:
    taxonomy = TAXONOMY_BY_CATEGORY[category]  # type: ignore[index]
    return FindingReportItem(
        test_id="TC-001",
        test_title="Unauthorized refund test",
        category=category,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        confidence="high",
        summary="Refund executed without server-side authorization.",
        taxonomy=taxonomy,
        evaluator="deterministic",
        evidence=["tool=request_refund authz=missing state_changed=True"],
        tool_call_ids=["tc_abc123"],
        limitations=[],
        remediation=REMEDIATION_BY_CATEGORY[category],  # type: ignore[index]
    )


@pytest.mark.unit
class TestSarifExport:
    def test_sarif_version_and_schema(self):
        report = _make_minimal_report()
        sarif = build_sarif(report)
        assert sarif["version"] == "2.1.0"
        assert "schema" in sarif["$schema"]

    def test_sarif_has_one_run(self):
        report = _make_minimal_report()
        sarif = build_sarif(report)
        assert len(sarif["runs"]) == 1

    def test_sarif_tool_name_is_agentshield(self):
        report = _make_minimal_report()
        sarif = build_sarif(report)
        driver = sarif["runs"][0]["tool"]["driver"]
        assert driver["name"] == "AgentShield"

    def test_sarif_rule_ids_are_stable(self):
        """Rule IDs must match the stable registry — never change between versions."""
        finding = _make_finding(category="unauthorized_tool_execution", verdict="confirmed")
        report = _make_minimal_report(findings=[finding])
        sarif = build_sarif(report)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = {r["id"] for r in rules}
        assert "agentshield/unauthorized-tool-execution" in rule_ids

    def test_sarif_confirmed_finding_maps_to_fail_level_error(self):
        finding = _make_finding(verdict="confirmed", severity="critical")
        report = _make_minimal_report(findings=[finding])
        sarif = build_sarif(report)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1
        result = results[0]
        assert result["kind"] == "fail"
        assert result["level"] == "error"

    def test_sarif_no_violation_maps_to_pass(self):
        finding = _make_finding(verdict="no_violation_observed", severity="info")
        report = _make_minimal_report(findings=[finding], overall_verdict="no_violation_observed")
        sarif = build_sarif(report)
        results = sarif["runs"][0]["results"]
        assert results[0]["kind"] == "pass"

    def test_sarif_suspected_maps_to_open_warning(self):
        finding = _make_finding(verdict="suspected", severity="medium")
        report = _make_minimal_report(findings=[finding], overall_verdict="suspected")
        sarif = build_sarif(report)
        result = sarif["runs"][0]["results"][0]
        assert result["kind"] == "open"
        assert result["level"] == "warning"

    def test_sarif_no_source_locations_fabricated(self):
        """Source locations must NOT appear in results when not known."""
        finding = _make_finding()
        report = _make_minimal_report(findings=[finding])
        sarif = build_sarif(report)
        for result in sarif["runs"][0]["results"]:
            assert "locations" not in result

    def test_sarif_rules_deduplicated_by_category(self):
        """Two findings of the same category must produce only one rule."""
        f1 = _make_finding(category="unauthorized_tool_execution", verdict="confirmed")
        f2 = _make_finding(category="unauthorized_tool_execution", verdict="suspected")
        report = _make_minimal_report(findings=[f1, f2])
        sarif = build_sarif(report)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        ids = [r["id"] for r in rules]
        assert ids.count("agentshield/unauthorized-tool-execution") == 1

    def test_sarif_empty_findings_still_emits_all_rules(self):
        """When there are no findings, all 6 rules are still included for CI visibility."""
        report = _make_minimal_report(findings=[], overall_verdict="no_violation_observed")
        sarif = build_sarif(report)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = {r["id"] for r in rules}
        for expected_id in _RULE_IDS.values():
            assert expected_id in rule_ids, f"Missing rule: {expected_id}"

    def test_sarif_scan_metadata_in_run_properties(self):
        report = _make_minimal_report()
        sarif = build_sarif(report)
        props = sarif["runs"][0]["properties"]
        assert props["scan_id"] == "scan_sarif_test"
        assert props["confirmed_count"] == 1

    def test_sarif_message_includes_evidence(self):
        finding = _make_finding()
        report = _make_minimal_report(findings=[finding])
        sarif = build_sarif(report)
        message = sarif["runs"][0]["results"][0]["message"]["text"]
        assert "CONFIRMED" in message
        assert "evidence" in message.lower() or "Evidence" in message

    def test_write_sarif_report_creates_file(self, tmp_path):
        finding = _make_finding()
        report = _make_minimal_report(findings=[finding])
        out = tmp_path / "report.sarif"
        result_path = write_sarif_report(report, out)
        assert result_path.exists()
        loaded = json.loads(result_path.read_text())
        assert loaded["version"] == "2.1.0"
        assert len(loaded["runs"][0]["results"]) == 1

    def test_sarif_is_valid_json(self):
        finding = _make_finding(verdict="confirmed")
        report = _make_minimal_report(findings=[finding])
        sarif = build_sarif(report)
        # Must serialize without errors
        serialized = json.dumps(sarif)
        reloaded = json.loads(serialized)
        assert reloaded["version"] == "2.1.0"
