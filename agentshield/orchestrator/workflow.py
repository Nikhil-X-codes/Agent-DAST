"""LangGraph-based multi-turn scan orchestrator with strict budget termination."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from agentshield.evaluation.evaluator import SecurityEvaluator, _aggregate_verdict
from agentshield.evaluation.models import EvaluationPolicy
from agentshield.harnesses.base import BaseHarness, SessionContext, TargetCapabilities
from agentshield.harnesses.exceptions import TargetUnauthorizedError
from agentshield.orchestrator.models import ScanBudget, ScanReport, TerminationReason
from agentshield.test_engine.engine import SecurityTestEngine
from agentshield.test_engine.models import (
    SecurityTestCase,
    TestCaseExecution,
    TestPlan,
    TestSuiteConfig,
)
from agentshield.tracing.collector import TraceCollector
from agentshield.tracing.models import ScanSessionRecord
from agentshield.tracing.store import TraceStore


class _OrchestratorState(TypedDict):
    scan_id: str
    started_at: str
    started_monotonic: float
    capabilities: TargetCapabilities | None
    plan: TestPlan | None
    current_index: int
    current_test: SecurityTestCase | None
    current_session: SessionContext | None
    turns_in_current: int
    total_turns_executed: int
    total_tokens_used: int
    executions: list[TestCaseExecution]
    termination_reason: TerminationReason
    final_report: ScanReport | None


class ScanOrchestrator:
    """LangGraph workflow orchestrating bounded, multi-turn security scans."""

    def __init__(
        self,
        harness: BaseHarness,
        policy: EvaluationPolicy,
        *,
        budget: ScanBudget | None = None,
        suite_config: TestSuiteConfig | None = None,
        test_engine: SecurityTestEngine | None = None,
        evaluator: SecurityEvaluator | None = None,
        trace_store: TraceStore | None = None,
        use_llm_judge: bool = False,
    ) -> None:
        if not harness.config.authorized:
            raise TargetUnauthorizedError(
                f"Target '{harness.config.target_name}' is not authorized for scanning."
            )
        self.harness = harness
        self.policy = policy
        self.budget = budget or ScanBudget()
        effective_max_tests = min(
            (suite_config.max_tests if suite_config else self.budget.max_tests),
            self.budget.max_tests,
        )
        self.suite_config = (suite_config or TestSuiteConfig()).model_copy(
            update={"max_tests": effective_max_tests}
        )
        self.test_engine = test_engine or SecurityTestEngine()
        self.evaluator = evaluator or SecurityEvaluator()
        self._external_store = trace_store
        self._active_store: TraceStore | None = None
        self._collector: TraceCollector | None = None
        self.use_llm_judge = use_llm_judge
        self._graph = self._compile_graph()

    def _compile_graph(self):
        graph = StateGraph(_OrchestratorState)
        graph.add_node("initialize_scan", self._initialize_scan)
        graph.add_node("select_test", self._select_test)
        graph.add_node("execute_target", self._execute_target)
        graph.add_node("collect_and_evaluate", self._collect_and_evaluate)
        graph.add_node("generate_report", self._generate_report)

        graph.set_entry_point("initialize_scan")
        graph.add_conditional_edges(
            "initialize_scan",
            self._route_after_init,
            {
                "select_test": "select_test",
                "generate_report": "generate_report",
            },
        )
        graph.add_edge("select_test", "execute_target")
        graph.add_edge("execute_target", "collect_and_evaluate")
        graph.add_conditional_edges(
            "collect_and_evaluate",
            self._more_tests_within_budget,
            {
                "select_test": "select_test",
                "generate_report": "generate_report",
            },
        )
        graph.add_edge("generate_report", END)
        return graph.compile()

    def _initialize_scan(self, state: _OrchestratorState) -> dict[str, object]:
        assert self._active_store is not None
        capabilities = self.harness.discover_capabilities()
        plan = self.test_engine.build_plan(capabilities, self.policy, config=self.suite_config)
        self._active_store.ensure_session(
            ScanSessionRecord(
                scan_id=state["scan_id"],
                target_name=capabilities.target_name,
                target_type=capabilities.target_type,
                observability=capabilities.observability,
                telemetry_available=capabilities.telemetry_available,
                metadata={"orchestrator": "langgraph", "tests_planned": len(plan.test_cases)},
            )
        )
        reason: TerminationReason = "no_tests_planned" if not plan.test_cases else "all_tests_completed"
        return {
            "capabilities": capabilities,
            "plan": plan,
            "termination_reason": reason,
        }

    def _route_after_init(self, state: _OrchestratorState) -> Literal["select_test", "generate_report"]:
        plan = state["plan"]
        if plan is None or not plan.test_cases:
            return "generate_report"
        return "select_test"

    def _select_test(self, state: _OrchestratorState) -> dict[str, object]:
        plan = state["plan"]
        assert plan is not None
        test_case = plan.test_cases[state["current_index"]]
        session = SessionContext(authenticated_subject=self.policy.authenticated_subject)
        return {
            "current_test": test_case,
            "current_session": session,
            "turns_in_current": 0,
        }

    def _execute_target(self, state: _OrchestratorState) -> dict[str, object]:
        assert self._collector is not None
        test_case = state["current_test"]
        session = state["current_session"]
        assert test_case is not None and session is not None

        turns = (test_case.turns or [test_case.prompt])[: self.budget.max_turns_per_test]
        tokens_used = state["total_tokens_used"]
        turns_executed = state["total_turns_executed"]
        turns_in_current = 0

        for turn_prompt in turns:
            if tokens_used >= self.budget.max_total_tokens:
                break
            if (time.monotonic() - state["started_monotonic"]) >= self.budget.max_duration_seconds:
                break

            result = self._collector.trace_execution(
                self.harness,
                turn_prompt,
                scan_id=state["scan_id"],
                session=session,
            )
            turns_in_current += 1
            turns_executed += 1
            if result.usage is not None:
                tokens_used += result.usage.total_tokens

        return {
            "turns_in_current": turns_in_current,
            "total_turns_executed": turns_executed,
            "total_tokens_used": tokens_used,
        }

    def _collect_and_evaluate(self, state: _OrchestratorState) -> dict[str, object]:
        assert self._active_store is not None
        test_case = state["current_test"]
        session = state["current_session"]
        assert test_case is not None and session is not None

        report = self.evaluator.evaluate_trace(
            self._active_store,
            state["scan_id"],
            self.policy,
            session_id=session.session_id,
            use_llm_judge=self.use_llm_judge,
        )
        execution = TestCaseExecution(
            test_case=test_case,
            scan_id=state["scan_id"],
            report=report,
        )
        return {
            "executions": [*state["executions"], execution],
            "current_index": state["current_index"] + 1,
        }

    def _more_tests_within_budget(self, state: _OrchestratorState) -> Literal["select_test", "generate_report"]:
        plan = state["plan"]
        assert plan is not None

        if len(state["executions"]) >= self.budget.max_tests:
            return "generate_report"
        if state["total_tokens_used"] >= self.budget.max_total_tokens:
            return "generate_report"
        if (time.monotonic() - state["started_monotonic"]) >= self.budget.max_duration_seconds:
            return "generate_report"
        if state["current_index"] >= len(plan.test_cases):
            return "generate_report"
        return "select_test"

    def _determine_termination_reason(self, state: _OrchestratorState) -> TerminationReason:
        plan = state["plan"]
        if plan is None or not plan.test_cases:
            return "no_tests_planned"
        if state["total_tokens_used"] >= self.budget.max_total_tokens:
            return "token_budget_exhausted"
        if (time.monotonic() - state["started_monotonic"]) >= self.budget.max_duration_seconds:
            return "duration_budget_exceeded"
        if state["current_index"] < len(plan.test_cases) and len(state["executions"]) >= self.budget.max_tests:
            return "max_tests_reached"
        return "all_tests_completed"

    def _generate_report(self, state: _OrchestratorState) -> dict[str, object]:
        caps = state["capabilities"]
        plan = state["plan"]
        assert caps is not None and plan is not None

        executions = state["executions"]
        all_findings = [f for ex in executions for f in ex.report.findings]
        confirmed_count = sum(1 for ex in executions if ex.report.overall_verdict == "confirmed")
        inconclusive_count = sum(1 for ex in executions if ex.report.overall_verdict == "inconclusive")
        reason = self._determine_termination_reason(state)

        report = ScanReport(
            scan_id=state["scan_id"],
            target_name=caps.target_name,
            target_type=caps.target_type,
            observability=caps.observability,
            telemetry_available=caps.telemetry_available,
            started_at=state["started_at"],
            completed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            termination_reason=reason,
            overall_verdict=_aggregate_verdict(all_findings),
            tests_planned=len(plan.test_cases),
            tests_executed=len(executions),
            total_turns_executed=state["total_turns_executed"],
            total_tokens_used=state["total_tokens_used"],
            budget=self.budget,
            plan=plan,
            executions=executions,
            confirmed_findings_count=confirmed_count,
            inconclusive_count=inconclusive_count,
            suite_config=self.suite_config,
        )
        return {"final_report": report, "termination_reason": reason}

    def run(
        self,
        *,
        scan_id: str | None = None,
        report_dir: str | Path | None = None,
    ) -> ScanReport:
        """Execute the full bounded LangGraph scan workflow and optionally write HTML/JSON reports."""
        active_scan_id = scan_id or f"scan_{uuid.uuid4().hex[:8]}"
        owns_store = self._external_store is None
        self._active_store = self._external_store or TraceStore(":memory:")
        self._collector = TraceCollector(self._active_store)

        initial_state: _OrchestratorState = {
            "scan_id": active_scan_id,
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "started_monotonic": time.monotonic(),
            "capabilities": None,
            "plan": None,
            "current_index": 0,
            "current_test": None,
            "current_session": None,
            "turns_in_current": 0,
            "total_turns_executed": 0,
            "total_tokens_used": 0,
            "executions": [],
            "termination_reason": "all_tests_completed",
            "final_report": None,
        }

        try:
            final_state = self._graph.invoke(initial_state)
            report = final_state["final_report"]
            assert report is not None
            if report_dir is not None:
                from agentshield.reporting.html_report import write_html_report
                from agentshield.reporting.json_report import write_json_report
                from agentshield.reporting.models import build_audit_report

                audit_data = build_audit_report(report, trace_store=self._active_store)
                out_dir = Path(report_dir)
                write_html_report(audit_data, out_dir / f"{active_scan_id}.html")
                write_json_report(audit_data, out_dir / f"{active_scan_id}.json")
            return report
        finally:
            if owns_store and self._active_store is not None:
                self._active_store.close()
            self._active_store = None
            self._collector = None
