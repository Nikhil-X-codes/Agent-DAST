"""Unified context-aware security test engine orchestrating plan -> validate -> execute -> evaluate."""

from __future__ import annotations

import uuid

from agentshield.evaluation.evaluator import SecurityEvaluator
from agentshield.evaluation.models import EvaluationPolicy
from agentshield.harnesses.base import BaseHarness, SessionContext, TargetCapabilities
from agentshield.test_engine.generator import TestVariationGenerator
from agentshield.test_engine.models import (
    SecurityTestCase,
    TestCaseExecution,
    TestPlan,
    TestSuiteConfig,
    TestSuiteResult,
)
from agentshield.test_engine.planner import ContextAwareTestPlanner
from agentshield.test_engine.validator import TestScopeValidator
from agentshield.tracing.collector import TraceCollector
from agentshield.tracing.store import TraceStore


class SecurityTestEngine:
    """Plans, validates, and executes context-aware security test suites against isolated targets."""

    def __init__(
        self,
        *,
        planner: ContextAwareTestPlanner | None = None,
        validator: TestScopeValidator | None = None,
        generator: TestVariationGenerator | None = None,
        evaluator: SecurityEvaluator | None = None,
    ) -> None:
        self.planner = planner or ContextAwareTestPlanner()
        self.validator = validator or TestScopeValidator()
        self.generator = generator
        self.evaluator = evaluator or SecurityEvaluator()

    def build_plan(
        self,
        capabilities: TargetCapabilities,
        policy: EvaluationPolicy,
        *,
        config: TestSuiteConfig | None = None,
    ) -> TestPlan:
        """Select relevant categories, generate bounded variations, and validate against scope/budget."""
        suite_config = config or TestSuiteConfig()
        plan = self.planner.plan_static_cases(capabilities, policy)
        proposed_cases: list[SecurityTestCase] = list(plan.test_cases)

        if suite_config.use_llm_variations and self.generator is not None:
            for category in plan.selected_categories:
                proposed_cases.extend(
                    self.generator.generate_for_category(
                        category,
                        capabilities=capabilities,
                        policy=policy,
                        config=suite_config,
                    )
                )

        accepted, rejected = self.validator.validate(
            proposed_cases,
            capabilities=capabilities,
            config=suite_config,
        )
        return plan.model_copy(
            update={
                "test_cases": accepted,
                "rejected_cases": rejected,
            }
        )

    def run_suite(
        self,
        harness: BaseHarness,
        policy: EvaluationPolicy,
        *,
        config: TestSuiteConfig | None = None,
        trace_store: TraceStore | None = None,
        scan_prefix: str | None = None,
    ) -> TestSuiteResult:
        """Discover target capabilities, build & validate test plan, execute, and evaluate."""
        capabilities = harness.discover_capabilities()
        plan = self.build_plan(capabilities, policy, config=config)

        active_store = trace_store or TraceStore(":memory:")
        owns_store = trace_store is None
        collector = TraceCollector(active_store)
        prefix = scan_prefix or f"suite_{uuid.uuid4().hex[:6]}"

        executions: list[TestCaseExecution] = []
        confirmed_count = 0
        inconclusive_count = 0

        try:
            for idx, case in enumerate(plan.test_cases, start=1):
                scan_id = f"{prefix}_{idx:02d}_{case.test_id}"
                session = SessionContext(authenticated_subject=policy.authenticated_subject)
                collector.trace_execution(harness, case.prompt, scan_id=scan_id, session=session)
                report = self.evaluator.evaluate_trace(active_store, scan_id, policy, session_id=session.session_id)
                executions.append(TestCaseExecution(test_case=case, scan_id=scan_id, report=report))
                if report.overall_verdict == "confirmed":
                    confirmed_count += 1
                elif report.overall_verdict == "inconclusive":
                    inconclusive_count += 1
        finally:
            if owns_store:
                active_store.close()

        return TestSuiteResult(
            plan=plan,
            executions=executions,
            confirmed_findings_count=confirmed_count,
            inconclusive_count=inconclusive_count,
        )
