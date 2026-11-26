"""Unit tests for regression replay store and helpers (Phase 8 add-on)."""

from __future__ import annotations

import pytest

from agentshield.reporting.models import FindingReportItem, REMEDIATION_BY_CATEGORY
from agentshield.evaluation.models import TAXONOMY_BY_CATEGORY
from agentshield.reporting.regression import (
    RegressionRecord,
    RegressionStore,
    create_regressions_from_findings,
    record_replay_result,
)


def _make_finding(
    test_id: str = "TC-001",
    category: str = "unauthorized_tool_execution",
    verdict: str = "confirmed",
    severity: str = "critical",
) -> FindingReportItem:
    taxonomy = TAXONOMY_BY_CATEGORY[category]  # type: ignore[index]
    return FindingReportItem(
        test_id=test_id,
        test_title=f"Test {test_id}",
        category=category,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        confidence="high",
        summary="Unauthorized refund executed.",
        taxonomy=taxonomy,
        evaluator="deterministic",
        evidence=["tool=request_refund authz=missing"],
        tool_call_ids=["tc_xyz"],
        limitations=[],
        remediation=REMEDIATION_BY_CATEGORY[category],  # type: ignore[index]
    )


@pytest.mark.unit
class TestCreateRegressionsFromFindings:
    def test_confirmed_finding_creates_regression(self):
        findings = [_make_finding(verdict="confirmed")]
        recs = create_regressions_from_findings("scan_001", findings)
        assert len(recs) == 1
        assert recs[0].original_verdict == "confirmed"
        assert recs[0].status == "open"
        assert recs[0].scan_id == "scan_001"

    def test_suspected_included_by_default(self):
        findings = [_make_finding(verdict="suspected")]
        recs = create_regressions_from_findings("scan_001", findings)
        assert len(recs) == 1
        assert recs[0].original_verdict == "suspected"

    def test_suspected_excluded_when_flag_false(self):
        findings = [_make_finding(verdict="suspected")]
        recs = create_regressions_from_findings("scan_001", findings, include_suspected=False)
        assert len(recs) == 0

    def test_inconclusive_never_creates_regression(self):
        findings = [_make_finding(verdict="inconclusive")]
        recs = create_regressions_from_findings("scan_001", findings)
        assert len(recs) == 0

    def test_no_violation_never_creates_regression(self):
        findings = [_make_finding(verdict="no_violation_observed")]
        recs = create_regressions_from_findings("scan_001", findings)
        assert len(recs) == 0

    def test_multiple_confirmed_creates_multiple_regressions(self):
        findings = [
            _make_finding("TC-001", verdict="confirmed"),
            _make_finding("TC-002", verdict="confirmed"),
        ]
        recs = create_regressions_from_findings("scan_001", findings)
        assert len(recs) == 2
        test_ids = {r.test_id for r in recs}
        assert "TC-001" in test_ids
        assert "TC-002" in test_ids

    def test_evidence_preserved_in_metadata(self):
        findings = [_make_finding(verdict="confirmed")]
        recs = create_regressions_from_findings("scan_001", findings)
        assert "evidence" in recs[0].metadata
        assert len(recs[0].metadata["evidence"]) > 0


@pytest.mark.unit
class TestRecordReplayResult:
    def test_replay_no_violation_marks_fixed(self):
        rec = RegressionRecord(
            scan_id="scan_001",
            test_id="TC-001",
            test_title="Test",
            category="unauthorized_tool_execution",
            original_verdict="confirmed",
            original_severity="critical",
            original_summary="Unauthorized refund.",
        )
        updated = record_replay_result(
            rec,
            replay_verdict="no_violation_observed",
            replay_summary="Guard denied the refund.",
        )
        assert updated.status == "fixed"
        assert updated.replay_verdict == "no_violation_observed"
        assert updated.replayed_at is not None

    def test_replay_confirmed_marks_regressed(self):
        rec = RegressionRecord(
            scan_id="scan_001",
            test_id="TC-001",
            test_title="Test",
            category="data_isolation_failure",
            original_verdict="confirmed",
            original_severity="high",
            original_summary="Cross-tenant breach.",
        )
        updated = record_replay_result(
            rec,
            replay_verdict="confirmed",
            replay_summary="Still breaches cross-tenant.",
        )
        assert updated.status == "regressed"

    def test_open_record_has_no_replay(self):
        rec = RegressionRecord(
            scan_id="scan_001",
            test_id="TC-001",
            test_title="Test",
            category="unauthorized_tool_execution",
            original_verdict="confirmed",
            original_severity="critical",
            original_summary="Unauthorized refund.",
        )
        assert rec.status == "open"
        assert rec.replay_verdict is None


@pytest.mark.unit
class TestRegressionStore:
    def test_save_and_load_by_scan(self):
        with RegressionStore() as store:
            rec = RegressionRecord(
                scan_id="scan_001",
                test_id="TC-001",
                test_title="Test",
                category="unauthorized_tool_execution",
                original_verdict="confirmed",
                original_severity="critical",
                original_summary="Unauthorized refund.",
            )
            store.save(rec)
            loaded = store.load_by_scan("scan_001")
            assert len(loaded) == 1
            assert loaded[0].regression_id == rec.regression_id

    def test_load_open_returns_only_open(self):
        with RegressionStore() as store:
            open_rec = RegressionRecord(
                scan_id="scan_001",
                test_id="TC-001",
                test_title="Open Test",
                category="unauthorized_tool_execution",
                original_verdict="confirmed",
                original_severity="critical",
                original_summary="Unauthorized refund.",
                status="open",
            )
            fixed_rec = RegressionRecord(
                scan_id="scan_001",
                test_id="TC-002",
                test_title="Fixed Test",
                category="data_isolation_failure",
                original_verdict="confirmed",
                original_severity="high",
                original_summary="Cross-tenant breach.",
                status="fixed",
            )
            store.save(open_rec)
            store.save(fixed_rec)
            open_records = store.load_open()
            assert len(open_records) == 1
            assert open_records[0].test_id == "TC-001"

    def test_save_then_update_replay_result(self):
        with RegressionStore() as store:
            rec = RegressionRecord(
                scan_id="scan_001",
                test_id="TC-001",
                test_title="Test",
                category="unauthorized_tool_execution",
                original_verdict="confirmed",
                original_severity="critical",
                original_summary="Unauthorized refund.",
            )
            store.save(rec)
            updated = record_replay_result(
                rec,
                replay_verdict="no_violation_observed",
                replay_summary="Guard fixed it.",
            )
            store.save(updated)
            loaded = store.load_all()
            assert len(loaded) == 1
            assert loaded[0].status == "fixed"
            assert loaded[0].replay_verdict == "no_violation_observed"

    def test_different_scans_isolated(self):
        with RegressionStore() as store:
            for scan_id in ["scan_A", "scan_B"]:
                rec = RegressionRecord(
                    scan_id=scan_id,
                    test_id="TC-001",
                    test_title="Test",
                    category="excessive_agency",
                    original_verdict="confirmed",
                    original_severity="high",
                    original_summary="Excessive agency.",
                )
                store.save(rec)
            assert len(store.load_by_scan("scan_A")) == 1
            assert len(store.load_by_scan("scan_B")) == 1
            assert len(store.load_all()) == 2


@pytest.mark.unit
class TestReplayExecution:
    def test_create_and_replay_regression_vulnerable_to_hardened(self):
        import json
        import httpx

        from agentshield.evaluation.models import EvaluationPolicy
        from agentshield.harnesses import LangGraphHarness, TargetConfig
        from agentshield.orchestrator import ScanBudget, ScanOrchestrator
        from agentshield.providers import GroqProvider, ProviderConfig
        from agentshield.reporting.regression import (
            create_regressions_from_scan_report,
            replay_scan_regressions,
        )
        from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB
        from agentshield.test_engine.models import TestSuiteConfig

        def make_provider() -> GroqProvider:
            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content)
                last_msg = body["messages"][-1]
                if last_msg["role"] == "user" and "ord_2001" in (last_msg.get("content") or ""):
                    return httpx.Response(
                        200,
                        json={
                            "model": "openai/gpt-oss-20b",
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [
                                            {
                                                "id": "call_reg_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "request_refund",
                                                    "arguments": json.dumps(
                                                        {"order_id": "ord_2001", "reason": "override"}
                                                    ),
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": "tool_calls",
                                }
                            ],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "model": "openai/gpt-oss-20b",
                        "choices": [{"message": {"role": "assistant", "content": "Handled."}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                )

            return GroqProvider(
                ProviderConfig(api_key="test-key", model="openai/gpt-oss-20b", max_retries=0),
                transport=httpx.MockTransport(handler),
            )

        policy = EvaluationPolicy(
            authenticated_subject="cust_alice",
            foreign_identifiers=["ord_2001", "cust_bob"],
        )

        with SupportSandboxDB() as db, RegressionStore() as reg_store:
            vuln_agent = CustomerSupportAgent(
                make_provider(), db, variant="vulnerable", authenticated_customer_id="cust_alice"
            )
            vuln_harness = LangGraphHarness(vuln_agent, TargetConfig(target_name="support-vuln", authorized=True))
            scan_report = ScanOrchestrator(
                vuln_harness,
                policy,
                budget=ScanBudget(max_tests=1, max_turns_per_test=1),
                suite_config=TestSuiteConfig(categories=["unauthorized_tool_execution"], max_tests=1),
            ).run()
            assert scan_report.overall_verdict == "confirmed"

            recs = create_regressions_from_scan_report(scan_report)
            assert len(recs) >= 1
            for rec in recs:
                assert rec.metadata.get("prompt")
                reg_store.save(rec)

            # Replay against hardened variant -> should now report fixed
            hardened_agent = CustomerSupportAgent(
                make_provider(), db, variant="hardened", authenticated_customer_id="cust_alice"
            )
            hardened_harness = LangGraphHarness(
                hardened_agent, TargetConfig(target_name="support-hardened", authorized=True)
            )
            replayed = replay_scan_regressions(reg_store, scan_report.scan_id, hardened_harness, policy)
            assert len(replayed) == len(recs)
            assert all(r.status == "fixed" for r in replayed)
            assert all(r.replay_verdict == "no_violation_observed" for r in replayed)
