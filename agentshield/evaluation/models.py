"""Pydantic models, policy definitions, and OWASP/MITRE mappings for Phase 5 evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Verdict = Literal["confirmed", "suspected", "inconclusive", "no_violation_observed"]
Severity = Literal["critical", "high", "medium", "low", "info"]
SecurityCategory = Literal[
    "sensitive_information_disclosure",
    "unauthorized_tool_execution",
    "excessive_agency",
    "data_isolation_failure",
    "resource_consumption",
    "indirect_prompt_injection",
]


class TaxonomyMapping(BaseModel):
    """OWASP LLM Top 10 and MITRE ATLAS taxonomy references."""

    owasp_llm: str
    mitre_atlas: str


TAXONOMY_BY_CATEGORY: dict[SecurityCategory, TaxonomyMapping] = {
    "sensitive_information_disclosure": TaxonomyMapping(
        owasp_llm="LLM02:2025 Sensitive Information Disclosure",
        mitre_atlas="AML.T0024 Exfiltration via ML Inference API",
    ),
    "unauthorized_tool_execution": TaxonomyMapping(
        owasp_llm="LLM06:2025 Excessive Agency",
        mitre_atlas="AML.T0051.000 LLM Prompt Injection",
    ),
    "excessive_agency": TaxonomyMapping(
        owasp_llm="LLM06:2025 Excessive Agency",
        mitre_atlas="AML.T0051.000 LLM Prompt Injection",
    ),
    "data_isolation_failure": TaxonomyMapping(
        owasp_llm="LLM02:2025 Sensitive Information Disclosure",
        mitre_atlas="AML.T0024 Exfiltration via ML Inference API",
    ),
    "resource_consumption": TaxonomyMapping(
        owasp_llm="LLM10:2025 Unbounded Consumption",
        mitre_atlas="AML.T0029 Denial of ML Service",
    ),
    "indirect_prompt_injection": TaxonomyMapping(
        owasp_llm="LLM01:2025 Prompt Injection",
        mitre_atlas="AML.T0051.001 Indirect Prompt Injection",
    ),
}


class EvaluationPolicy(BaseModel):
    """Policy rules and synthetic boundaries for evaluating an agent execution."""

    authenticated_subject: str | None = None
    allowed_tools: list[str] | None = None
    forbidden_tools: list[str] = Field(default_factory=list)
    canary_secrets: list[str] = Field(default_factory=list)
    foreign_identifiers: list[str] = Field(default_factory=list)
    untrusted_input_present: bool = False
    max_total_tokens: int | None = None
    max_llm_steps: int | None = None
    max_tool_calls: int | None = None


class SecurityFinding(BaseModel):
    """Evidence-based security evaluation finding."""

    category: SecurityCategory
    verdict: Verdict
    severity: Severity
    title: str
    summary: str
    taxonomy: TaxonomyMapping
    evaluator: Literal["deterministic", "llm_judge"] = "deterministic"
    evidence: list[str] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Aggregated evaluation output for a single target execution."""

    scan_id: str | None = None
    target_name: str
    target_type: str
    observability: str
    telemetry_available: bool
    findings: list[SecurityFinding] = Field(default_factory=list)
    overall_verdict: Verdict = "no_violation_observed"
    limitations: list[str] = Field(default_factory=list)
