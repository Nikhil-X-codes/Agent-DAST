"""Pydantic models for Phase 7 multi-turn scan orchestration and budgets."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from agentshield.evaluation.models import Verdict
from agentshield.test_engine.models import TestCaseExecution, TestPlan, TestSuiteConfig

TerminationReason = Literal[
    "all_tests_completed",
    "max_tests_reached",
    "token_budget_exhausted",
    "duration_budget_exceeded",
    "no_tests_planned",
]


class ScanBudget(BaseModel):
    """Hard bounds ensuring every scan terminates deterministically."""

    max_tests: int = Field(default=6, ge=1, le=50)
    max_turns_per_test: int = Field(default=3, ge=1, le=10)
    max_total_tokens: int = Field(default=20_000, ge=1)
    max_duration_seconds: float = Field(default=120.0, gt=0)


class ScanReport(BaseModel):
    """Final aggregated report produced by the ScanOrchestrator LangGraph workflow."""

    scan_id: str
    target_name: str
    target_type: str
    observability: str
    telemetry_available: bool
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    termination_reason: TerminationReason
    overall_verdict: Verdict
    tests_planned: int
    tests_executed: int
    total_turns_executed: int
    total_tokens_used: int
    budget: ScanBudget
    plan: TestPlan
    executions: list[TestCaseExecution] = Field(default_factory=list)
    confirmed_findings_count: int = 0
    inconclusive_count: int = 0
    suite_config: TestSuiteConfig = Field(default_factory=TestSuiteConfig)
