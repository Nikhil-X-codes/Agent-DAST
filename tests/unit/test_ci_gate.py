"""Unit tests for CI failure gate functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentshield.evaluation.models import TAXONOMY_BY_CATEGORY
from agentshield.reporting.models import (
    REMEDIATION_BY_CATEGORY,
    AuditReportData,
    FindingReportItem,
    has_confirmed_critical_or_high,
)


def _make_report(findings: list[FindingReportItem], overall: str = "confirmed") -> AuditReportData:
    return AuditReportData(
        scan_id="scan_ci_test",
        target_name="support-vulnerable",
        target_type="langgraph",
        observability="instrumented",
        telemetry_available=True,
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:01:00Z",
        termination_reason="all_tests_completed",
        overall_verdict=overall,  # type: ignore[arg-type]
        total_tests=3,
        completed_tests=3,
        incomplete_tests=0,
        confirmed_count=sum(1 for f in findings if f.verdict == "confirmed"),
        suspected_count=0,
        inconclusive_count=0,
        no_violation_count=sum(1 for f in findings if f.verdict == "no_violation_observed"),
        total_tokens_used=1500,
        total_turns_executed=3,
        total_latency_ms=820.0,
        avg_latency_ms=273.3,
        findings=findings,
    )


def _finding(verdict: str, severity: str, category: str = "unauthorized_tool_execution") -> FindingReportItem:
    taxonomy = TAXONOMY_BY_CATEGORY[category]  # type: ignore[index]
    return FindingReportItem(
        test_id="TC-001",
        test_title="Test",
        category=category,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        confidence="high",
        summary="Test finding.",
        taxonomy=taxonomy,
        evaluator="deterministic",
        evidence=[],
        tool_call_ids=[],
        limitations=[],
        remediation=REMEDIATION_BY_CATEGORY[category],  # type: ignore[index]
    )


def _has_confirmed_critical_or_high(report: AuditReportData) -> bool:
    """CI gate check: return True if any finding is confirmed with critical or high severity."""
    return has_confirmed_critical_or_high(report)


@pytest.mark.unit
class TestCIGate:
    def test_confirmed_critical_triggers_gate(self):
        report = _make_report([_finding("confirmed", "critical")])
        assert _has_confirmed_critical_or_high(report) is True

    def test_confirmed_high_triggers_gate(self):
        report = _make_report([_finding("confirmed", "high")])
        assert _has_confirmed_critical_or_high(report) is True

    def test_confirmed_medium_does_not_trigger_gate(self):
        report = _make_report([_finding("confirmed", "medium")])
        assert _has_confirmed_critical_or_high(report) is False

    def test_suspected_critical_does_not_trigger_gate(self):
        """Gate only triggers on CONFIRMED findings — suspected is not enough."""
        report = _make_report([_finding("suspected", "critical")])
        assert _has_confirmed_critical_or_high(report) is False

    def test_no_violation_does_not_trigger_gate(self):
        report = _make_report([_finding("no_violation_observed", "info")], overall="no_violation_observed")
        assert _has_confirmed_critical_or_high(report) is False

    def test_empty_findings_does_not_trigger_gate(self):
        report = _make_report([], overall="no_violation_observed")
        assert _has_confirmed_critical_or_high(report) is False

    def test_mixed_findings_triggers_if_any_confirmed_critical(self):
        report = _make_report([
            _finding("no_violation_observed", "info"),
            _finding("confirmed", "critical"),
            _finding("suspected", "high"),
        ])
        assert _has_confirmed_critical_or_high(report) is True

    def test_json_report_loads_correctly_for_gate_check(self, tmp_path):
        """Verify the gate works when loading a saved JSON report."""
        from agentshield.reporting.json_report import write_json_report

        report = _make_report([_finding("confirmed", "critical")])
        json_path = tmp_path / "scan.json"
        write_json_report(report, json_path)

        # Simulate what a CI script does: load JSON, re-instantiate, check gate
        raw = json.loads(json_path.read_text())
        loaded = AuditReportData.model_validate(raw)
        assert _has_confirmed_critical_or_high(loaded) is True

    def test_clean_report_does_not_trigger_gate_from_json(self, tmp_path):
        from agentshield.reporting.json_report import write_json_report

        report = _make_report([_finding("no_violation_observed", "info")], overall="no_violation_observed")
        json_path = tmp_path / "clean_scan.json"
        write_json_report(report, json_path)

        raw = json.loads(json_path.read_text())
        loaded = AuditReportData.model_validate(raw)
        assert _has_confirmed_critical_or_high(loaded) is False

    def test_cli_report_gate_fails_on_confirmed_critical_and_exports_sarif(self, tmp_path):
        from typer.testing import CliRunner

        from agentshield.cli import app
        from agentshield.reporting.json_report import write_json_report

        runner = CliRunner()
        report = _make_report([_finding("confirmed", "critical")])
        json_path = tmp_path / "failing.json"
        write_json_report(report, json_path)

        result = runner.invoke(
            app,
            ["report", str(json_path), "--sarif", "--out-dir", str(tmp_path), "--fail-on-confirmed"],
        )
        assert result.exit_code == 1
        assert (tmp_path / f"{report.scan_id}.sarif").exists()

    def test_cli_report_gate_passes_on_clean_report(self, tmp_path):
        from typer.testing import CliRunner

        from agentshield.cli import app
        from agentshield.reporting.json_report import write_json_report

        runner = CliRunner()
        report = _make_report([_finding("no_violation_observed", "info")], overall="no_violation_observed")
        json_path = tmp_path / "passing.json"
        write_json_report(report, json_path)

        result = runner.invoke(
            app,
            ["report", str(json_path), "--sarif", "--out-dir", str(tmp_path), "--fail-on-confirmed"],
        )
        assert result.exit_code == 0

    def test_cli_manifest_runs_offline_without_api_keys(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from agentshield.cli import app

        for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        out_file = tmp_path / "manifest.json"
        result = CliRunner().invoke(app, ["manifest", "--variant", "hardened", "--out", str(out_file)])
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data["tools"]) == 5
