"""HTTP target harness for auditing local or remote authorized HTTP agents."""

from __future__ import annotations

import time
from typing import Any

import httpx

from agentshield.harnesses.base import (
    BaseHarness,
    DiscoveredTool,
    ObservabilityMode,
    SessionContext,
    StandardizedExecutionResult,
    TargetCapabilities,
    TargetConfig,
    ToolCallTrace,
)
from agentshield.harnesses.exceptions import HarnessConfigError, TargetExecutionError
from agentshield.providers.base import TokenUsage


class HTTPHarness(BaseHarness):
    """Harness for auditing HTTP-hosted AI agents (black-box or instrumented)."""

    target_type = "http"

    def __init__(
        self,
        base_url: str,
        config: TargetConfig,
        *,
        invoke_path: str = "/invoke",
        capabilities_path: str | None = "/capabilities",
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        if not base_url:
            raise HarnessConfigError("HTTPHarness requires a non-empty base_url.")
        self.base_url = base_url.rstrip("/")
        self.invoke_path = invoke_path if invoke_path.startswith("/") else f"/{invoke_path}"
        if capabilities_path is None:
            self.capabilities_path = None
        else:
            self.capabilities_path = (
                capabilities_path if capabilities_path.startswith("/") else f"/{capabilities_path}"
            )
        self._headers = dict(headers or {})
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def discover_capabilities(self) -> TargetCapabilities:
        tools: list[DiscoveredTool] = list(self.config.explicit_tools)
        permissions: list[str] = list(self.config.declared_permissions)
        observability: ObservabilityMode = "black_box"
        telemetry_available = False
        session_handling = "stateless"
        metadata: dict[str, Any] = {"base_url": self.base_url}

        if self.capabilities_path is not None:
            url = f"{self.base_url}{self.capabilities_path}"
            try:
                resp = self._client.get(url, headers=self._headers)
            except httpx.HTTPError as exc:
                raise TargetExecutionError(
                    f"Capability discovery failed for HTTP target '{self.config.target_name}': {type(exc).__name__}"
                ) from exc

            if resp.status_code == 200:
                data = resp.json()
                for raw_tool in data.get("tools") or []:
                    tools.append(
                        DiscoveredTool(
                            name=str(raw_tool.get("name", "")),
                            description=str(raw_tool.get("description", "")),
                            input_schema=dict(raw_tool.get("input_schema") or raw_tool.get("parameters") or {}),
                            permissions=list(raw_tool.get("permissions") or []),
                            source="http_metadata",
                        )
                    )
                permissions.extend(str(p) for p in (data.get("declared_permissions") or []))
                telemetry_available = bool(data.get("telemetry_available", False))
                observability = "instrumented" if telemetry_available else "black_box"
                session_handling = str(data.get("session_handling") or "stateless")
                metadata["discovered_from"] = url
            elif resp.status_code != 404:
                raise TargetExecutionError(
                    f"HTTP capability endpoint '{url}' returned status {resp.status_code}."
                )

        return TargetCapabilities(
            target_type="http",
            target_name=self.config.target_name,
            observability=observability,
            telemetry_available=telemetry_available,
            session_handling=session_handling,  # type: ignore[arg-type]
            tools=tools,
            declared_permissions=permissions,
            metadata=metadata,
        )

    def execute(
        self,
        prompt: str,
        *,
        session: SessionContext | None = None,
    ) -> StandardizedExecutionResult:
        active_session = session or SessionContext()
        url = f"{self.base_url}{self.invoke_path}"
        payload = {
            "prompt": prompt,
            "session_id": active_session.session_id,
            "authenticated_subject": active_session.authenticated_subject,
        }

        start = time.perf_counter()
        try:
            resp = self._client.post(url, headers=self._headers, json=payload)
        except httpx.HTTPError as exc:
            raise TargetExecutionError(
                f"HTTP target '{self.config.target_name}' request failed: {type(exc).__name__}"
            ) from exc
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        if resp.status_code >= 400:
            raise TargetExecutionError(
                f"HTTP target '{self.config.target_name}' returned status {resp.status_code}."
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise TargetExecutionError(
                f"HTTP target '{self.config.target_name}' returned non-JSON response."
            ) from exc

        response_text = str(
            data.get("response_text") or data.get("final_response") or data.get("response") or ""
        )

        raw_traces = data.get("execution_trace") or data.get("tool_executions")
        traces: list[ToolCallTrace] | None = None
        telemetry_available = False
        observability: ObservabilityMode = "black_box"
        if isinstance(raw_traces, list):
            telemetry_available = True
            observability = "instrumented"
            traces = [
                ToolCallTrace(
                    tool_call_id=item.get("tool_call_id"),
                    tool_name=str(item.get("tool_name", "")),
                    arguments=dict(item.get("arguments") or {}),
                    authorized=item.get("authorized"),
                    status=item.get("status"),
                    state_changed=item.get("state_changed"),
                    latency_ms=item.get("latency_ms"),
                    output=item.get("output"),
                )
                for item in raw_traces
                if isinstance(item, dict)
            ]

        usage = None
        if isinstance(data.get("usage"), dict):
            u = data["usage"]
            usage = TokenUsage(
                prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
                completion_tokens=int(u.get("completion_tokens", 0) or 0),
                total_tokens=int(u.get("total_tokens", 0) or 0),
            )

        return StandardizedExecutionResult(
            target_type="http",
            target_name=self.config.target_name,
            observability=observability,
            telemetry_available=telemetry_available,
            session=active_session,
            prompt=prompt,
            response_text=response_text,
            provider=data.get("provider"),
            model=data.get("model"),
            latency_ms=latency_ms,
            usage=usage,
            execution_trace=traces,
            raw_metadata={
                k: v
                for k, v in data.items()
                if k not in {"response_text", "final_response", "response", "execution_trace", "tool_executions", "usage"}
            },
        )
