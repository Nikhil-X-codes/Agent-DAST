"""Context-aware security test planner based on TargetCapabilities and EvaluationPolicy."""

from __future__ import annotations

from agentshield.evaluation.models import EvaluationPolicy, SecurityCategory
from agentshield.harnesses.base import TargetCapabilities
from agentshield.test_engine.models import SecurityTestCase, TestPlan
from agentshield.test_engine.templates import build_static_templates, is_mutating_tool, is_read_tool


class ContextAwareTestPlanner:
    """Selects relevant security test categories and static regression templates for a target."""

    def select_categories(
        self,
        capabilities: TargetCapabilities,
        policy: EvaluationPolicy,
    ) -> tuple[list[SecurityCategory], dict[str, str]]:
        selected: list[SecurityCategory] = []
        rationale: dict[str, str] = {}

        tools = capabilities.tools
        mutating_tools = [t.name for t in tools if is_mutating_tool(t)]
        read_tools = [t.name for t in tools if is_read_tool(t)]

        def _add(cat: SecurityCategory, reason: str) -> None:
            if cat not in selected:
                selected.append(cat)
                rationale[cat] = reason

        if mutating_tools:
            _add(
                "unauthorized_tool_execution",
                f"Target exposes state-changing/mutating tools: {', '.join(mutating_tools)}.",
            )

        if policy.allowed_tools is not None or policy.forbidden_tools:
            _add(
                "excessive_agency",
                "Policy defines explicit allowed_tools or forbidden_tools boundaries.",
            )

        if read_tools or policy.foreign_identifiers or not tools:
            _add(
                "data_isolation_failure",
                "Target handles data lookup or tenant-scoped records."
                if tools
                else "Black-box target requires tenant/data isolation probing.",
            )

        if read_tools or policy.canary_secrets or not tools:
            _add(
                "sensitive_information_disclosure",
                "Target retrieves documents/records or has configured canary secrets.",
            )

        if policy.untrusted_input_present or any(
            kw in t.name.lower() for t in tools for kw in ("faq", "doc", "search", "fetch", "web")
        ):
            _add(
                "indirect_prompt_injection",
                "Target processes retrieved/untrusted text content.",
            )

        if (
            policy.max_total_tokens is not None
            or policy.max_llm_steps is not None
            or policy.max_tool_calls is not None
        ):
            _add(
                "resource_consumption",
                "Policy specifies explicit token or execution step budgets.",
            )

        return selected, rationale

    def plan_static_cases(
        self,
        capabilities: TargetCapabilities,
        policy: EvaluationPolicy,
    ) -> TestPlan:
        """Build an initial TestPlan containing contextually selected static regression tests."""
        categories, rationale = self.select_categories(capabilities, policy)
        cases: list[SecurityTestCase] = []
        for category in categories:
            cases.extend(build_static_templates(category, tools=capabilities.tools, policy=policy))

        return TestPlan(
            target_name=capabilities.target_name,
            target_type=capabilities.target_type,
            selected_categories=categories,
            selection_rationale=rationale,
            test_cases=cases,
        )
