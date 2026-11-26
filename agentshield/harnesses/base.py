"""BaseHarness interface and normalized capability/execution models."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentshield.harnesses.exceptions import HarnessConfigError, TargetUnauthorizedError
from agentshield.providers.base import TokenUsage

ObservabilityMode = Literal["black_box", "instrumented"]
TargetType = Literal["python", "http", "langgraph", "langchain", "mcp"]


class DiscoveredTool(BaseModel):
    """Normalized metadata for a tool exposed or declared by a target agent."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)
    source: Literal["explicit_schema", "mcp_metadata", "instrumented", "http_metadata"] = "explicit_schema"


class SessionContext(BaseModel):
    """Normalized session and identity context for a target execution."""

    session_id: str = Field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:10]}")
    authenticated_subject: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetCapabilities(BaseModel):
    """Normalized capability discovery report for a target agent."""

    target_type: TargetType
    target_name: str
    observability: ObservabilityMode
    telemetry_available: bool
    session_handling: Literal["none", "stateless", "session_bound"] = "none"
    tools: list[DiscoveredTool] = Field(default_factory=list)
    declared_permissions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallTrace(BaseModel):
    """Normalized trace entry for an observed tool call (only when telemetry is available)."""

    tool_call_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    authorized: bool | None = None
    status: str | None = None
    state_changed: bool | None = None
    latency_ms: float | None = None
    output: dict[str, Any] | str | None = None


class StandardizedExecutionResult(BaseModel):
    """Normalized execution result across Python, HTTP, LangGraph/LangChain, and MCP targets."""

    target_type: TargetType
    target_name: str
    observability: ObservabilityMode
    telemetry_available: bool
    session: SessionContext
    prompt: str
    response_text: str
    provider: str | None = None
    model: str | None = None
    latency_ms: float
    usage: TokenUsage | None = None
    execution_trace: list[ToolCallTrace] | None = None
    error: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class TargetConfig(BaseModel):
    """Explicit target configuration required for live security auditing."""

    target_name: str
    authorized: bool = False
    live_scan: bool = True
    allow_mock_fallback: bool = False
    timeout_seconds: float = Field(default=30.0, gt=0)
    explicit_tools: list[DiscoveredTool] = Field(default_factory=list)
    declared_permissions: list[str] = Field(default_factory=list)


class BaseHarness(ABC):
    """Abstract base class for auditing heterogeneous AI agent targets."""

    target_type: TargetType

    def __init__(self, config: TargetConfig) -> None:
        if config.live_scan and config.allow_mock_fallback:
            raise HarnessConfigError(
                f"Target '{config.target_name}' cannot enable allow_mock_fallback during a live scan."
            )
        if config.live_scan and not config.authorized:
            raise TargetUnauthorizedError(
                f"Target '{config.target_name}' requires explicit authorization (authorized=True) for live scans."
            )
        self.config = config

    @abstractmethod
    def discover_capabilities(self) -> TargetCapabilities:
        """Discover declared tools, schemas, permissions, session handling, and telemetry support."""

    @abstractmethod
    def execute(
        self,
        prompt: str,
        *,
        session: SessionContext | None = None,
    ) -> StandardizedExecutionResult:
        """Execute a single prompt against the target and return a normalized result."""

    def close(self) -> None:
        """Release any resources held by the harness."""

    def __enter__(self) -> BaseHarness:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
