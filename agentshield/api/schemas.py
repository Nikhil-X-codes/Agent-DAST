"""Pydantic request and response schemas for the AgentShield FastAPI service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentshield.evaluation.models import EvaluationPolicy, SecurityCategory, Verdict
from agentshield.orchestrator.models import ScanBudget
from agentshield.reporting.models import AuditReportData, FindingReportItem
from agentshield.targets.support.tools import AgentVariant

ScanStatus = Literal["queued", "running", "completed", "cancelled", "failed"]
TargetKind = Literal["support_demo", "http", "mcp"]


class CreateScanRequest(BaseModel):
    """Request payload to initiate an authorized security scan."""

    target_name: str = Field(default="support-target", min_length=1, max_length=100)
    target_type: TargetKind = "support_demo"
    authorized: bool = Field(description="Explicit target authorization flag; must be true.")
    support_variant: AgentVariant = "hardened"
    target_url: str | None = Field(default=None, description="Required for http or mcp targets.")
    authenticated_customer_id: str = "cust_alice"
    policy: EvaluationPolicy | None = None
    budget: ScanBudget = Field(default_factory=lambda: ScanBudget(max_tests=3, max_turns_per_test=2))
    use_llm_variations: bool = False
    use_llm_judge: bool = False


class ScanStatusResponse(BaseModel):
    """Current status and summary metadata of a scan job."""

    scan_id: str
    status: ScanStatus
    target_name: str
    target_type: str
    created_at: str
    updated_at: str
    overall_verdict: Verdict | None = None
    tests_planned: int = 0
    tests_executed: int = 0
    total_tokens_used: int = 0
    termination_reason: str | None = None
    error: str | None = None


class ScanFindingsResponse(BaseModel):
    """Findings retrieved for a completed or partial scan."""

    scan_id: str
    status: ScanStatus
    overall_verdict: Verdict | None = None
    findings: list[FindingReportItem] = Field(default_factory=list)


class TestSuiteDescriptor(BaseModel):
    """Metadata describing an available AgentShield security test category."""

    category: SecurityCategory
    description: str
    owasp_llm: str
    mitre_atlas: str
    applicable_to: str


class TestSuitesResponse(BaseModel):
    """List of available security test suites."""

    suites: list[TestSuiteDescriptor]


class ReportResponse(BaseModel):
    """JSON wrapper for a completed scan report."""

    scan_id: str
    status: ScanStatus
    report: AuditReportData
