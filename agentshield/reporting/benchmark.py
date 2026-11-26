"""Comparative benchmarking between vulnerable and hardened real agent variants."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentshield.evaluation.models import EvaluationPolicy
from agentshield.harnesses.base import TargetConfig
from agentshield.harnesses.langgraph_harness import LangGraphHarness
from agentshield.orchestrator.models import ScanBudget
from agentshield.orchestrator.workflow import ScanOrchestrator
from agentshield.providers.base import BaseLLMProvider
from agentshield.providers.router import FreeProviderRouter
from agentshield.reporting.models import AuditReportData, build_audit_report
from agentshield.targets.support.agent import CustomerSupportAgent
from agentshield.targets.support.db import SupportSandboxDB
from agentshield.targets.support.tools import AgentVariant
from agentshield.test_engine.models import TestSuiteConfig
from agentshield.tracing.store import TraceStore


class VariantBenchmarkSummary(BaseModel):
    """Measured benchmark metrics for a single agent variant across repeated trials."""

    variant: AgentVariant
    trials: int
    total_tests_executed: int
    total_confirmed_findings: int
    total_inconclusive: int
    confirmed_violation_rate: float
    total_tokens_used: int
    avg_latency_ms: float
    reports: list[AuditReportData] = Field(default_factory=list)


class BenchmarkComparison(BaseModel):
    """Side-by-side measured comparison of vulnerable vs. hardened variants."""

    trials: int
    vulnerable: VariantBenchmarkSummary
    hardened: VariantBenchmarkSummary
    confirmed_reduction_count: int


class BenchmarkRunner:
    """Runs equivalent security scans against vulnerable and hardened real agents."""

    def __init__(
        self,
        provider: BaseLLMProvider | FreeProviderRouter,
        policy: EvaluationPolicy,
        *,
        budget: ScanBudget | None = None,
        suite_config: TestSuiteConfig | None = None,
        authenticated_customer_id: str = "cust_alice",
    ) -> None:
        self.provider = provider
        self.policy = policy
        self.budget = budget or ScanBudget(max_tests=3, max_turns_per_test=2)
        self.suite_config = suite_config or TestSuiteConfig(max_tests=self.budget.max_tests)
        self.authenticated_customer_id = authenticated_customer_id

    def run(self, *, trials: int = 1) -> BenchmarkComparison:
        if trials < 1:
            raise ValueError("trials must be >= 1")

        vuln_summary = self._run_variant("vulnerable", trials=trials)
        hardened_summary = self._run_variant("hardened", trials=trials)

        return BenchmarkComparison(
            trials=trials,
            vulnerable=vuln_summary,
            hardened=hardened_summary,
            confirmed_reduction_count=(
                vuln_summary.total_confirmed_findings - hardened_summary.total_confirmed_findings
            ),
        )

    def _run_variant(self, variant: AgentVariant, *, trials: int) -> VariantBenchmarkSummary:
        reports: list[AuditReportData] = []
        total_tests = 0
        total_confirmed = 0
        total_inconclusive = 0
        total_tokens = 0
        latencies: list[float] = []

        for trial in range(1, trials + 1):
            with SupportSandboxDB(":memory:") as db, TraceStore(":memory:") as store:
                agent = CustomerSupportAgent(
                    self.provider,
                    db,
                    variant=variant,
                    authenticated_customer_id=self.authenticated_customer_id,
                )
                harness = LangGraphHarness(
                    agent,
                    TargetConfig(target_name=f"support-{variant}", authorized=True),
                )
                orchestrator = ScanOrchestrator(
                    harness,
                    self.policy,
                    budget=self.budget,
                    suite_config=self.suite_config,
                    trace_store=store,
                )
                scan_report = orchestrator.run(scan_id=f"bench_{variant}_t{trial}")
                audit_report = build_audit_report(scan_report, trace_store=store)
                reports.append(audit_report)

                total_tests += audit_report.completed_tests
                total_confirmed += audit_report.confirmed_count
                total_inconclusive += audit_report.inconclusive_count
                total_tokens += audit_report.total_tokens_used
                if audit_report.avg_latency_ms > 0:
                    latencies.append(audit_report.avg_latency_ms)

        violation_rate = round(total_confirmed / total_tests, 4) if total_tests > 0 else 0.0
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        return VariantBenchmarkSummary(
            variant=variant,
            trials=trials,
            total_tests_executed=total_tests,
            total_confirmed_findings=total_confirmed,
            total_inconclusive=total_inconclusive,
            confirmed_violation_rate=violation_rate,
            total_tokens_used=total_tokens,
            avg_latency_ms=avg_latency,
            reports=reports,
        )
