"""Unified evidence-based security policy evaluator."""

from __future__ import annotations

from typing import Sequence

from agentshield.evaluation.deterministic import (
    evaluate_data_isolation,
    evaluate_excessive_agency,
    evaluate_indirect_prompt_injection,
    evaluate_resource_consumption,
    evaluate_sensitive_information_disclosure,
    evaluate_unauthorized_tool_execution,
)
from agentshield.evaluation.llm_judge import SemanticLLMJudge
from agentshield.evaluation.models import (
    EvaluationPolicy,
    EvaluationReport,
    SecurityFinding,
    Verdict,
)
from agentshield.harnesses.base import StandardizedExecutionResult
from agentshield.tracing.models import LLMTraceEvent, ToolTraceEvent
from agentshield.tracing.store import TraceStore

_VERDICT_PRIORITY: dict[Verdict, int] = {
    "confirmed": 4,
    "suspected": 3,
    "inconclusive": 2,
    "no_violation_observed": 1,
}


def _aggregate_verdict(findings: Sequence[SecurityFinding]) -> Verdict:
    if not findings:
        return "no_violation_observed"
    return max(findings, key=lambda f: _VERDICT_PRIORITY[f.verdict]).verdict


class SecurityEvaluator:
    """Evaluates recorded or live target executions using deterministic rules and optional LLM judge."""

    def __init__(self, *, llm_judge: SemanticLLMJudge | None = None) -> None:
        self.llm_judge = llm_judge

    def evaluate_trace(
        self,
        store: TraceStore,
        scan_id: str,
        policy: EvaluationPolicy,
        *,
        session_id: str | None = None,
        use_llm_judge: bool = False,
    ) -> EvaluationReport:
        """Evaluate a persisted scan session from SQLite TraceStore."""
        session = store.get_session(scan_id)
        if session is None:
            raise ValueError(f"Scan session '{scan_id}' not found in TraceStore.")

        llm_events = store.get_llm_events(scan_id, session_id=session_id)
        tool_events = store.get_tool_events(scan_id, session_id=session_id)
        response_text = ""
        prompt = ""
        if llm_events:
            prompt = str(llm_events[0].prompt_input or "")
            for ev in reversed(llm_events):
                if ev.response_output:
                    response_text = ev.response_output
                    break

        return self._evaluate(
            scan_id=scan_id,
            target_name=session.target_name,
            target_type=session.target_type,
            observability=session.observability,
            telemetry_available=session.telemetry_available,
            prompt=prompt,
            response_text=response_text,
            llm_events=llm_events,
            tool_events=tool_events,
            policy=policy,
            use_llm_judge=use_llm_judge,
        )

    def evaluate_result(
        self,
        result: StandardizedExecutionResult,
        policy: EvaluationPolicy,
        *,
        scan_id: str | None = None,
        use_llm_judge: bool = False,
    ) -> EvaluationReport:
        """Evaluate an in-memory StandardizedExecutionResult directly."""
        variant = result.raw_metadata.get("variant")
        tool_events: list[ToolTraceEvent] = []
        for trace in result.execution_trace or []:
            authz = (
                "denied"
                if trace.authorized is False
                else ("missing" if (trace.authorized is True and variant == "vulnerable") else ("allowed" if trace.authorized else "unavailable"))
            )
            tool_events.append(
                ToolTraceEvent(
                    scan_id=scan_id or "in_memory",
                    session_id=result.session.session_id,
                    tool_call_id=trace.tool_call_id,
                    tool_name=trace.tool_name,
                    arguments=trace.arguments,
                    result=trace.output,
                    authorization=authz,  # type: ignore[arg-type]
                    execution_status="completed" if trace.status == "success" else (trace.status or "unknown"),
                    state_changed=trace.state_changed,
                    latency_ms=trace.latency_ms,
                )
            )

        llm_events = [
            LLMTraceEvent(
                scan_id=scan_id or "in_memory",
                session_id=result.session.session_id,
                step=1,
                provider=result.provider,
                model=result.model,
                prompt_input=result.prompt,
                response_output=result.response_text,
                latency_ms=result.latency_ms,
                usage=result.usage,
                telemetry_available=result.telemetry_available,
                error=result.error,
            )
        ]

        return self._evaluate(
            scan_id=scan_id,
            target_name=result.target_name,
            target_type=result.target_type,
            observability=result.observability,
            telemetry_available=result.telemetry_available,
            prompt=result.prompt,
            response_text=result.response_text,
            llm_events=llm_events,
            tool_events=tool_events,
            policy=policy,
            use_llm_judge=use_llm_judge,
        )

    def _evaluate(
        self,
        *,
        scan_id: str | None,
        target_name: str,
        target_type: str,
        observability: str,
        telemetry_available: bool,
        prompt: str,
        response_text: str,
        llm_events: Sequence[LLMTraceEvent],
        tool_events: Sequence[ToolTraceEvent],
        policy: EvaluationPolicy,
        use_llm_judge: bool,
    ) -> EvaluationReport:
        findings: list[SecurityFinding] = [
            evaluate_unauthorized_tool_execution(
                telemetry_available=telemetry_available,
                tool_events=tool_events,
                policy=policy,
            ),
            evaluate_sensitive_information_disclosure(
                response_text=response_text,
                tool_events=tool_events,
                policy=policy,
            ),
            evaluate_excessive_agency(
                telemetry_available=telemetry_available,
                tool_events=tool_events,
                policy=policy,
            ),
            evaluate_data_isolation(
                telemetry_available=telemetry_available,
                response_text=response_text,
                tool_events=tool_events,
                policy=policy,
            ),
            evaluate_resource_consumption(
                llm_events=llm_events,
                tool_events=tool_events,
                policy=policy,
            ),
            evaluate_indirect_prompt_injection(
                telemetry_available=telemetry_available,
                tool_events=tool_events,
                policy=policy,
            ),
        ]

        if use_llm_judge and self.llm_judge is not None:
            findings.append(
                self.llm_judge.evaluate(
                    prompt=prompt,
                    response_text=response_text,
                    tool_events=tool_events,
                    policy=policy,
                )
            )

        limitations: list[str] = []
        for f in findings:
            for lim in f.limitations:
                if lim not in limitations:
                    limitations.append(lim)

        return EvaluationReport(
            scan_id=scan_id,
            target_name=target_name,
            target_type=target_type,
            observability=observability,
            telemetry_available=telemetry_available,
            findings=findings,
            overall_verdict=_aggregate_verdict(findings),
            limitations=limitations,
        )
