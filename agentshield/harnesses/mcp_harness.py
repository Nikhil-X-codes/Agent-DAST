"""Authorized MCP (Model Context Protocol) JSON-RPC target harness."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from agentshield.harnesses.base import (
    BaseHarness,
    DiscoveredTool,
    SessionContext,
    StandardizedExecutionResult,
    TargetCapabilities,
    TargetConfig,
    ToolCallTrace,
)
from agentshield.harnesses.exceptions import HarnessConfigError, TargetExecutionError


class MCPHarness(BaseHarness):
    """Harness for authorized MCP servers exposing JSON-RPC `tools/list` and `tools/call`."""

    target_type = "mcp"

    def __init__(
        self,
        endpoint_url: str,
        config: TargetConfig,
        *,
        default_tool: str | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        if not endpoint_url:
            raise HarnessConfigError("MCPHarness requires a non-empty endpoint_url.")
        self.endpoint_url = endpoint_url
        self.default_tool = default_tool
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds),
            transport=transport,
        )
        self._rpc_id = 0

    def close(self) -> None:
        self._client.close()

    def _rpc_call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._rpc_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._rpc_id,
            "method": method,
            "params": params or {},
        }
        try:
            resp = self._client.post(self.endpoint_url, headers=self._headers, json=payload)
        except httpx.HTTPError as exc:
            raise TargetExecutionError(
                f"MCP request '{method}' to '{self.config.target_name}' failed: {type(exc).__name__}"
            ) from exc

        if resp.status_code >= 400:
            raise TargetExecutionError(
                f"MCP endpoint '{self.endpoint_url}' returned HTTP {resp.status_code}."
            )

        data = resp.json()
        if "error" in data and data["error"] is not None:
            err = data["error"]
            raise TargetExecutionError(f"MCP error {err.get('code')}: {err.get('message')}")
        return dict(data.get("result") or {})

    def discover_capabilities(self) -> TargetCapabilities:
        result = self._rpc_call("tools/list")
        tools: list[DiscoveredTool] = list(self.config.explicit_tools)
        for raw_tool in result.get("tools") or []:
            tools.append(
                DiscoveredTool(
                    name=str(raw_tool.get("name", "")),
                    description=str(raw_tool.get("description", "")),
                    input_schema=dict(raw_tool.get("inputSchema") or {}),
                    permissions=list(raw_tool.get("permissions") or []),
                    source="mcp_metadata",
                )
            )

        return TargetCapabilities(
            target_type="mcp",
            target_name=self.config.target_name,
            observability="instrumented",
            telemetry_available=True,
            session_handling="session_bound",
            tools=tools,
            declared_permissions=list(self.config.declared_permissions),
            metadata={"endpoint_url": self.endpoint_url},
        )

    def execute(
        self,
        prompt: str,
        *,
        session: SessionContext | None = None,
    ) -> StandardizedExecutionResult:
        active_session = session or SessionContext()
        tool_name: str | None = self.default_tool
        arguments: dict[str, Any] = {"prompt": prompt}

        try:
            parsed = json.loads(prompt)
            if isinstance(parsed, dict) and "tool" in parsed:
                tool_name = str(parsed["tool"])
                arguments = dict(parsed.get("arguments") or {})
        except ValueError:
            pass

        if not tool_name:
            raise TargetExecutionError(
                "MCPHarness requires either a JSON prompt {'tool': ..., 'arguments': ...} or default_tool."
            )

        start = time.perf_counter()
        result = self._rpc_call(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
                "_meta": {
                    "session_id": active_session.session_id,
                    "authenticated_subject": active_session.authenticated_subject,
                },
            },
        )
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        is_error = bool(result.get("isError", False))
        content_items = result.get("content") or []
        text_parts = [
            str(item.get("text", ""))
            for item in content_items
            if isinstance(item, dict) and "text" in item
        ]
        response_text = "\n".join(text_parts) if text_parts else json.dumps(result)

        trace = ToolCallTrace(
            tool_name=tool_name,
            arguments=arguments,
            authorized=not is_error,
            status="error" if is_error else "success",
            output=response_text,
        )

        return StandardizedExecutionResult(
            target_type="mcp",
            target_name=self.config.target_name,
            observability="instrumented",
            telemetry_available=True,
            session=active_session,
            prompt=prompt,
            response_text=response_text,
            latency_ms=latency_ms,
            execution_trace=[trace],
        )
