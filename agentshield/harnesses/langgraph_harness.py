"""LangGraph / LangChain target harness with authorized instrumentation."""

from __future__ import annotations

import time
from typing import Any

from agentshield.harnesses.base import (
    BaseHarness,
    DiscoveredTool,
    SessionContext,
    StandardizedExecutionResult,
    TargetCapabilities,
    TargetConfig,
    TargetType,
    ToolCallTrace,
)
from agentshield.harnesses.exceptions import HarnessConfigError, TargetExecutionError
from agentshield.targets.support.agent import AgentRunResult, CustomerSupportAgent


def _is_instrumented_agent(agent: Any) -> bool:
    return isinstance(agent, CustomerSupportAgent) or (
        hasattr(agent, "executor")
        and hasattr(agent.executor, "tool_definitions")
        and hasattr(agent, "run")
        and hasattr(agent, "authenticated_customer_id")
        and hasattr(agent, "variant")
    )


class LangGraphHarness(BaseHarness):
    """Harness for instrumented AgentShield LangGraph agents, LangGraph graphs, and LangChain runnables."""

    target_type: TargetType = "langgraph"

    def __init__(
        self,
        agent: Any,
        config: TargetConfig,
        *,
        target_type: TargetType = "langgraph",
    ) -> None:
        super().__init__(config)
        if not (_is_instrumented_agent(agent) or hasattr(agent, "invoke")):
            raise HarnessConfigError(
                "LangGraphHarness requires an instrumented agent (e.g., CustomerSupportAgent/RAGAgent) "
                "or an object with an .invoke() method."
            )
        self._agent = agent
        self.target_type = target_type

    def discover_capabilities(self) -> TargetCapabilities:
        if _is_instrumented_agent(self._agent):
            tools = [
                DiscoveredTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.parameters,
                    permissions=[f"variant:{self._agent.variant}"],
                    source="instrumented",
                )
                for tool in self._agent.executor.tool_definitions
            ]
            tools.extend(self.config.explicit_tools)
            return TargetCapabilities(
                target_type=self.target_type,
                target_name=self.config.target_name,
                observability="instrumented",
                telemetry_available=True,
                session_handling="session_bound",
                tools=tools,
                declared_permissions=list(self.config.declared_permissions)
                or [f"authenticated_customer:{self._agent.authenticated_customer_id}"],
                metadata={"variant": self._agent.variant},
            )

        # Generic LangGraph / LangChain runnable: do not invent tools if none are explicitly supplied.
        return TargetCapabilities(
            target_type=self.target_type,
            target_name=self.config.target_name,
            observability="black_box" if not self.config.explicit_tools else "instrumented",
            telemetry_available=False,
            session_handling="stateless",
            tools=list(self.config.explicit_tools),
            declared_permissions=list(self.config.declared_permissions),
        )

    def execute(
        self,
        prompt: str,
        *,
        session: SessionContext | None = None,
    ) -> StandardizedExecutionResult:
        active_session = session or SessionContext()

        if _is_instrumented_agent(self._agent):
            if (
                active_session.authenticated_subject
                and active_session.authenticated_subject != self._agent.authenticated_customer_id
            ):
                # Re-bind only if explicitly requested via session context on a new run
                self._agent.authenticated_customer_id = active_session.authenticated_subject
                self._agent.executor.authenticated_customer_id = active_session.authenticated_subject
            elif active_session.authenticated_subject is None:
                active_session = active_session.model_copy(
                    update={"authenticated_subject": self._agent.authenticated_customer_id}
                )

            try:
                result: AgentRunResult = self._agent.run(prompt)
            except Exception as exc:
                raise TargetExecutionError(
                    f"LangGraph target '{self.config.target_name}' failed: {exc}"
                ) from exc

            traces = [
                ToolCallTrace(
                    tool_call_id=rec.tool_call_id,
                    tool_name=rec.tool_name,
                    arguments=rec.arguments,
                    authorized=rec.authorized,
                    status=rec.status,
                    state_changed=rec.state_changed,
                    latency_ms=rec.latency_ms,
                    output=rec.output,
                )
                for rec in result.tool_executions
            ]
            return StandardizedExecutionResult(
                target_type=self.target_type,
                target_name=self.config.target_name,
                observability="instrumented",
                telemetry_available=True,
                session=active_session,
                prompt=prompt,
                response_text=result.final_response,
                provider=result.provider,
                model=result.model,
                latency_ms=result.latency_ms,
                usage=result.usage,
                execution_trace=traces,
                raw_metadata={
                    "variant": result.variant,
                    "steps": result.steps,
                    "llm_turns": [t.model_dump() for t in result.llm_turns],
                },
            )

        # Generic LangChain / LangGraph runnable
        start = time.perf_counter()
        try:
            raw = self._agent.invoke({"input": prompt})
        except Exception as exc:
            raise TargetExecutionError(
                f"LangChain/LangGraph target '{self.config.target_name}' failed: {exc}"
            ) from exc
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        if isinstance(raw, dict):
            text = str(raw.get("output") or raw.get("response") or raw.get("content") or "")
        else:
            text = str(getattr(raw, "content", raw))

        return StandardizedExecutionResult(
            target_type=self.target_type,
            target_name=self.config.target_name,
            observability="black_box",
            telemetry_available=False,
            session=active_session,
            prompt=prompt,
            response_text=text,
            latency_ms=latency_ms,
            execution_trace=None,
        )
