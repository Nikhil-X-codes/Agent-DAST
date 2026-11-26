"""Pydantic models for the context-aware security test engine."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentshield.evaluation.models import EvaluationReport, SecurityCategory

TestSource = Literal["static_template", "llm_variation"]


class TestSuiteConfig(BaseModel):
    """Configuration and budget bounds for test planning, generation, and execution."""

    max_tests: int = Field(default=12, ge=1, le=50)
    max_variations_per_category: int = Field(default=2, ge=0, le=5)
    use_llm_variations: bool = False
    allow_external_urls: bool = False
    generator_temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class SecurityTestCase(BaseModel):
    """A single validated security test case with provenance and expected invariant."""

    test_id: str = Field(default_factory=lambda: f"tc_{uuid.uuid4().hex[:8]}")
    category: SecurityCategory
    title: str
    prompt: str
    turns: list[str] | None = None
    expected_invariant: str
    target_tools: list[str] = Field(default_factory=list)
    source: TestSource = "static_template"
    generator_provider: str | None = None
    generator_model: str | None = None
    generator_config: dict[str, Any] = Field(default_factory=dict)


class RejectedTestCase(BaseModel):
    """A generated or proposed test case rejected by scope/safety/budget validation."""

    title: str
    prompt: str
    category: SecurityCategory
    source: TestSource
    rejection_reason: str


class TestPlan(BaseModel):
    """Context-aware test plan derived from TargetCapabilities and EvaluationPolicy."""

    target_name: str
    target_type: str
    selected_categories: list[SecurityCategory] = Field(default_factory=list)
    selection_rationale: dict[str, str] = Field(default_factory=dict)
    test_cases: list[SecurityTestCase] = Field(default_factory=list)
    rejected_cases: list[RejectedTestCase] = Field(default_factory=list)


class TestCaseExecution(BaseModel):
    """Execution and evaluation outcome for a single SecurityTestCase."""

    test_case: SecurityTestCase
    scan_id: str
    report: EvaluationReport


class TestSuiteResult(BaseModel):
    """Aggregated result of planning and executing a context-aware security test suite."""

    plan: TestPlan
    executions: list[TestCaseExecution] = Field(default_factory=list)
    confirmed_findings_count: int = 0
    inconclusive_count: int = 0
