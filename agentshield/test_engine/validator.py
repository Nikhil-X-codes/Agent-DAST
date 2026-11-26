"""Scope, safety, and budget validator for proposed security test cases."""

from __future__ import annotations

import re

from agentshield.harnesses.base import TargetCapabilities
from agentshield.test_engine.models import RejectedTestCase, SecurityTestCase, TestSuiteConfig

_EXTERNAL_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DESTRUCTIVE_RE = re.compile(
    r"\b(rm\s+-rf|drop\s+table|truncate\s+table|delete\s+from|os\.system|subprocess\.|powershell\s+-enc|mkfs|shutdown)\b",
    re.IGNORECASE,
)
_REAL_CREDENTIAL_RE = re.compile(
    r"(Bearer\s+[A-Za-z0-9\-._~+/]+=*|\bgsk_[A-Za-z0-9]{16,}\b|\bAIza[0-9A-Za-z\-_]{20,}\b|\bsk-or-v1-[A-Za-z0-9]{16,}\b|\bsk-[A-Za-z0-9]{20,}\b)"
)


class TestScopeValidator:
    """Validates test cases against target scope, safety boundaries, and execution budgets."""

    def validate(
        self,
        cases: list[SecurityTestCase],
        *,
        capabilities: TargetCapabilities,
        config: TestSuiteConfig,
    ) -> tuple[list[SecurityTestCase], list[RejectedTestCase]]:
        accepted: list[SecurityTestCase] = []
        rejected: list[RejectedTestCase] = []
        known_tools = {t.name for t in capabilities.tools}

        for case in cases:
            reason = self._check_case(case, known_tools=known_tools, config=config)
            if reason is not None:
                rejected.append(
                    RejectedTestCase(
                        title=case.title,
                        prompt=case.prompt,
                        category=case.category,
                        source=case.source,
                        rejection_reason=reason,
                    )
                )
                continue

            if len(accepted) >= config.max_tests:
                rejected.append(
                    RejectedTestCase(
                        title=case.title,
                        prompt=case.prompt,
                        category=case.category,
                        source=case.source,
                        rejection_reason=f"Exceeds max_tests execution budget ({config.max_tests}).",
                    )
                )
                continue

            accepted.append(case)

        return accepted, rejected

    @staticmethod
    def _check_case(
        case: SecurityTestCase,
        *,
        known_tools: set[str],
        config: TestSuiteConfig,
    ) -> str | None:
        if not config.allow_external_urls and _EXTERNAL_URL_RE.search(case.prompt):
            return "Test prompt references an unauthorized external URL."

        if _DESTRUCTIVE_RE.search(case.prompt):
            return "Test prompt contains destructive or host-command patterns."

        if _REAL_CREDENTIAL_RE.search(case.prompt):
            return "Test prompt contains real credential or bearer token patterns."

        if case.source == "llm_variation" and known_tools and case.target_tools:
            unknown = [t for t in case.target_tools if t not in known_tools]
            if unknown:
                return f"LLM variation references undeclared out-of-scope tools: {', '.join(unknown)}."

        if not case.prompt.strip():
            return "Test prompt is empty."

        return None
