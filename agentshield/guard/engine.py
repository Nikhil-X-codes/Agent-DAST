"""PolicyEngine — deterministic tool-call authorization for AgentShield Guard."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel

from agentshield.guard.models import ApprovalToken, GuardPolicy

DenyReason = Literal[
    "forbidden_tool",
    "not_in_allowlist",
    "tenant_not_allowed",
    "ownership_violation",
    "approval_required",
    "approval_expired",
    "approval_mismatch",
    "session_call_limit_exceeded",
    "unknown_sensitive_action",
]


class GuardContext(BaseModel):
    """Typed input context for a single policy evaluation."""

    actor: str  # authenticated_subject / tenant ID — from application session, not LLM
    tool_name: str
    arguments: dict[str, Any]
    session_id: str
    approval_token: ApprovalToken | None = None
    calls_this_session: int = 0  # count of tool calls already executed in this session
    resolved_owner: str | None = None  # server-side resolved owner when ownership is indirect (e.g. order_id -> customer_id)


class PolicyDecision(BaseModel):
    """Result of a PolicyEngine evaluation."""

    allowed: bool
    deny_reason: DenyReason | None = None
    deny_detail: str | None = None
    tool_name: str
    actor: str
    session_id: str
    approval_required: bool = False
    arguments_hash: str | None = None


def _hash_arguments(arguments: dict[str, Any]) -> str:
    """Canonical SHA-256 hash of tool arguments (sorted keys, no whitespace)."""
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class PolicyEngine:
    """Deterministic, default-deny policy engine for AgentShield Guard.

    Evaluation order (fail-fast, default-deny):
    1. Forbidden tool → always deny
    2. Allowlist check (if configured) → deny if not in list
    3. Tenant allowlist check (if configured) → deny if actor not allowed
    4. Ownership check (if require_ownership) → deny if ownership_field mismatch
    5. Session call limit (if configured) → deny if exceeded
    6. Sensitive tool approval check → deny if no valid token
    7. Allow
    """

    def __init__(self, policy: GuardPolicy) -> None:
        self.policy = policy

    def evaluate(self, ctx: GuardContext) -> PolicyDecision:
        """Evaluate a tool-call context against the configured policy. Returns PolicyDecision."""
        p = self.policy
        args_hash = _hash_arguments(ctx.arguments)

        # 1. Forbidden tool — always deny
        if ctx.tool_name in p.forbidden_tools:
            return PolicyDecision(
                allowed=False,
                deny_reason="forbidden_tool",
                deny_detail=f"Tool '{ctx.tool_name}' is permanently forbidden by policy.",
                tool_name=ctx.tool_name,
                actor=ctx.actor,
                session_id=ctx.session_id,
                arguments_hash=args_hash,
            )

        # 2. Default-deny unknown tools when enabled
        if p.default_deny_unknown and p.known_tools is not None and ctx.tool_name not in p.known_tools:
            return PolicyDecision(
                allowed=False,
                deny_reason="unknown_sensitive_action",
                deny_detail=f"Tool '{ctx.tool_name}' is unknown and denied by default-deny policy.",
                tool_name=ctx.tool_name,
                actor=ctx.actor,
                session_id=ctx.session_id,
                arguments_hash=args_hash,
            )

        # 3. Allowlist check
        if p.allowed_tools is not None and ctx.tool_name not in p.allowed_tools:
            return PolicyDecision(
                allowed=False,
                deny_reason="not_in_allowlist",
                deny_detail=(
                    f"Tool '{ctx.tool_name}' is not in the allowed_tools list. "
                    f"Allowed: {p.allowed_tools}"
                ),
                tool_name=ctx.tool_name,
                actor=ctx.actor,
                session_id=ctx.session_id,
                arguments_hash=args_hash,
            )

        # 4. Tenant allowlist check
        if p.allowed_tenants is not None and ctx.actor not in p.allowed_tenants:
            return PolicyDecision(
                allowed=False,
                deny_reason="tenant_not_allowed",
                deny_detail=(
                    f"Actor '{ctx.actor}' is not in the allowed_tenants list."
                ),
                tool_name=ctx.tool_name,
                actor=ctx.actor,
                session_id=ctx.session_id,
                arguments_hash=args_hash,
            )

        # 5. Ownership check — server-side resolved owner or ownership_field argument must match actor
        if p.require_ownership:
            if ctx.resolved_owner is not None and ctx.resolved_owner != ctx.actor:
                return PolicyDecision(
                    allowed=False,
                    deny_reason="ownership_violation",
                    deny_detail=(
                        f"Tool '{ctx.tool_name}' target resource owner '{ctx.resolved_owner}' "
                        f"does not match authenticated actor '{ctx.actor}'."
                    ),
                    tool_name=ctx.tool_name,
                    actor=ctx.actor,
                    session_id=ctx.session_id,
                    arguments_hash=args_hash,
                )
            owner_in_args = ctx.arguments.get(p.ownership_field)
            if owner_in_args is not None and str(owner_in_args) != ctx.actor:
                return PolicyDecision(
                    allowed=False,
                    deny_reason="ownership_violation",
                    deny_detail=(
                        f"Tool '{ctx.tool_name}' argument '{p.ownership_field}' value "
                        f"'{owner_in_args}' does not match authenticated actor '{ctx.actor}'."
                    ),
                    tool_name=ctx.tool_name,
                    actor=ctx.actor,
                    session_id=ctx.session_id,
                    arguments_hash=args_hash,
                )

        # 5. Session call limit
        if p.max_calls_per_session is not None and ctx.calls_this_session >= p.max_calls_per_session:
            return PolicyDecision(
                allowed=False,
                deny_reason="session_call_limit_exceeded",
                deny_detail=(
                    f"Session '{ctx.session_id}' has reached the maximum tool calls "
                    f"per session ({p.max_calls_per_session})."
                ),
                tool_name=ctx.tool_name,
                actor=ctx.actor,
                session_id=ctx.session_id,
                arguments_hash=args_hash,
            )

        # 6. Sensitive tool — require a valid, bound, non-expired ApprovalToken
        if ctx.tool_name in p.sensitive_tools:
            token = ctx.approval_token
            if token is None:
                return PolicyDecision(
                    allowed=False,
                    deny_reason="approval_required",
                    deny_detail=(
                        f"Tool '{ctx.tool_name}' is classified as sensitive and requires "
                        "an explicit human-issued ApprovalToken. Model-generated approval "
                        "assertions are not accepted."
                    ),
                    tool_name=ctx.tool_name,
                    actor=ctx.actor,
                    session_id=ctx.session_id,
                    approval_required=True,
                    arguments_hash=args_hash,
                )
            if token.is_expired():
                return PolicyDecision(
                    allowed=False,
                    deny_reason="approval_expired",
                    deny_detail=(
                        f"ApprovalToken '{token.token_id}' for tool '{ctx.tool_name}' "
                        "has expired. A new human-issued approval is required."
                    ),
                    tool_name=ctx.tool_name,
                    actor=ctx.actor,
                    session_id=ctx.session_id,
                    approval_required=True,
                    arguments_hash=args_hash,
                )
            if not token.matches(ctx.tool_name, args_hash, ctx.actor):
                return PolicyDecision(
                    allowed=False,
                    deny_reason="approval_mismatch",
                    deny_detail=(
                        f"ApprovalToken '{token.token_id}' does not match the exact tool "
                        f"name, arguments hash, or actor for this call. Approval is bound "
                        "to a specific action and cannot be reused."
                    ),
                    tool_name=ctx.tool_name,
                    actor=ctx.actor,
                    session_id=ctx.session_id,
                    approval_required=True,
                    arguments_hash=args_hash,
                )

        # 7. Allow
        return PolicyDecision(
            allowed=True,
            tool_name=ctx.tool_name,
            actor=ctx.actor,
            session_id=ctx.session_id,
            arguments_hash=args_hash,
        )
