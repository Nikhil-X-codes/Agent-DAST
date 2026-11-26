"""Python callable target harness (black-box or explicitly instrumented)."""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable

from agentshield.harnesses.base import (
    BaseHarness,
    ObservabilityMode,
    SessionContext,
    StandardizedExecutionResult,
    TargetCapabilities,
    TargetConfig,
    ToolCallTrace,
)
from agentshield.harnesses.exceptions import HarnessConfigError, TargetExecutionError


class PythonCallableHarness(BaseHarness):
    """Harness for auditing local Python agent callables."""

    target_type = "python"

    def __init__(
        self,
        callable_fn: Callable[..., Any],
        config: TargetConfig,
        *,
        observability: ObservabilityMode = "black_box",
    ) -> None:
        super().__init__(config)
        if not callable(callable_fn):
            raise HarnessConfigError("PythonCallableHarness requires a callable target.")
        self._fn = callable_fn
        self._observability = observability
        self._sig = inspect.signature(callable_fn)

    def _supports_session_param(self) -> bool:
        return "session" in self._sig.parameters

    def discover_capabilities(self) -> TargetCapabilities:
        return TargetCapabilities(
            target_type="python",
            target_name=self.config.target_name,
            observability=self._observability,
            telemetry_available=(self._observability == "instrumented"),
            session_handling="session_bound" if self._supports_session_param() else "stateless",
            tools=list(self.config.explicit_tools),
            declared_permissions=list(self.config.declared_permissions),
            metadata={"callable_name": getattr(self._fn, "__name__", type(self._fn).__name__)},
        )

    def execute(
        self,
        prompt: str,
        *,
        session: SessionContext | None = None,
    ) -> StandardizedExecutionResult:
        active_session = session or SessionContext()
        start = time.perf_counter()
        try:
            if self._supports_session_param():
                raw = self._fn(prompt, session=active_session)
            else:
                raw = self._fn(prompt)
        except Exception as exc:
            raise TargetExecutionError(
                f"Python target '{self.config.target_name}' raised {type(exc).__name__}: {exc}"
            ) from exc

        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return self._normalize_output(raw, prompt=prompt, session=active_session, latency_ms=latency_ms)

    def _normalize_output(
        self,
        raw: Any,
        *,
        prompt: str,
        session: SessionContext,
        latency_ms: float,
    ) -> StandardizedExecutionResult:
        if isinstance(raw, str):
            return StandardizedExecutionResult(
                target_type="python",
                target_name=self.config.target_name,
                observability=self._observability,
                telemetry_available=False,
                session=session,
                prompt=prompt,
                response_text=raw,
                latency_ms=latency_ms,
                execution_trace=None,
            )

        if isinstance(raw, dict):
            response_text = str(raw.get("response_text") or raw.get("final_response") or raw.get("output") or "")
            raw_traces = raw.get("execution_trace") or raw.get("tool_executions")
            traces: list[ToolCallTrace] | None = None
            telemetry_available = False
            if self._observability == "instrumented" and isinstance(raw_traces, list):
                telemetry_available = True
                traces = [
                    ToolCallTrace(
                        tool_name=str(item.get("tool_name", "")),
                        arguments=dict(item.get("arguments") or {}),
                        authorized=item.get("authorized"),
                        status=item.get("status"),
                        output=item.get("output"),
                    )
                    for item in raw_traces
                    if isinstance(item, dict)
                ]

            return StandardizedExecutionResult(
                target_type="python",
                target_name=self.config.target_name,
                observability=self._observability,
                telemetry_available=telemetry_available,
                session=session,
                prompt=prompt,
                response_text=response_text,
                provider=raw.get("provider"),
                model=raw.get("model"),
                latency_ms=latency_ms,
                execution_trace=traces,
                raw_metadata={k: v for k, v in raw.items() if k not in {"response_text", "final_response", "execution_trace", "tool_executions"}},
            )

        return StandardizedExecutionResult(
            target_type="python",
            target_name=self.config.target_name,
            observability=self._observability,
            telemetry_available=False,
            session=session,
            prompt=prompt,
            response_text=str(raw),
            latency_ms=latency_ms,
            execution_trace=None,
        )
