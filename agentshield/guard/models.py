"""Typed Pydantic models for AgentShield Guard policy and approval tokens."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class GuardPolicy(BaseModel):
    """Policy rules for the AgentShield Guard runtime gateway.

    Fields:
        allowed_tools: If set, only these tools are permitted to execute.
        sensitive_tools: Tools that require an explicit ApprovalToken before execution.
        forbidden_tools: Tools that are always denied regardless of approval.
        allowed_tenants: If set, only tool calls on behalf of these tenant IDs are permitted.
        max_calls_per_session: Hard cap on total tool calls within a single session.
        require_ownership: When True, tool arguments must include a resource owned by authenticated_subject.
        ownership_field: Argument field name that carries the resource owner/customer ID.
    """

    allowed_tools: list[str] | None = None
    known_tools: list[str] | None = None
    default_deny_unknown: bool = False
    sensitive_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    allowed_tenants: list[str] | None = None
    max_calls_per_session: int | None = None
    require_ownership: bool = False
    ownership_field: str = "customer_id"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalToken(BaseModel):
    """Human-approval token bound to an exact tool call and actor.

    Design constraints (agent1.md Phase 7):
    - Must be bound to the exact tool_name + arguments hash (not any tool call).
    - Must expire after ttl_seconds.
    - Model-generated approval strings are NEVER accepted; only tokens issued
      by the application session are valid.
    - issued_by must come from the application session, never from LLM output.
    """

    token_id: str
    tool_name: str
    arguments_hash: str  # sha256 of canonicalized arguments JSON
    actor: str  # authenticated_subject / tenant who was approved
    issued_by: str  # e.g. "human_operator" or "session_service" — not "llm"
    issued_at: float = Field(default_factory=time.time)
    ttl_seconds: float = 300.0

    def is_expired(self) -> bool:
        return (time.time() - self.issued_at) > self.ttl_seconds

    def matches(self, tool_name: str, arguments_hash: str, actor: str) -> bool:
        return (
            self.tool_name == tool_name
            and self.arguments_hash == arguments_hash
            and self.actor == actor
            and not self.is_expired()
        )
