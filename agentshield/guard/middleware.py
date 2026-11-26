"""AgentShieldGuard — middleware wrapping SupportToolExecutor with policy enforcement.

Integrates the PolicyEngine with the existing SupportToolExecutor so every tool call
passes through the guard before reaching the sandboxed database tools.

Usage::

    from agentshield.guard import AgentShieldGuard, GuardPolicy
    from agentshield.targets.support.tools import SupportToolExecutor

    policy = GuardPolicy(sensitive_tools=["request_refund"], forbidden_tools=["admin_reset"])
    executor = SupportToolExecutor(db, variant="hardened", authenticated_customer_id="cust_alice")
    guard = AgentShieldGuard(executor, policy, session_id="sess_123", actor="cust_alice")

    record = guard.execute("request_refund", {"order_id": "ord_1001", "reason": "damaged"})
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from agentshield.guard.engine import GuardContext, PolicyDecision, PolicyEngine
from agentshield.guard.models import ApprovalToken, GuardPolicy
from agentshield.targets.support.tools import ToolExecutionRecord

if TYPE_CHECKING:
    from agentshield.tracing.store import TraceStore

logger = logging.getLogger(__name__)


class AgentShieldGuard:
    """Runtime policy gateway wrapping a SupportToolExecutor or compatible tool executor.

    - Every tool call passes through PolicyEngine before execution.
    - Denied calls are logged and returned as denied ToolExecutionRecord (not executed).
    - Approved calls are forwarded to the underlying executor.
    - Decision log is accessible via `self.decision_log` and optionally persisted to `TraceStore`.
    """

    def __init__(
        self,
        executor: Any,  # SupportToolExecutor or any object with .execute(tool_name, args, **kw)
        policy: GuardPolicy,
        *,
        session_id: str,
        actor: str,
        resource_owner_resolver: Callable[[str, dict[str, Any]], str | None] | None = None,
        trace_store: "TraceStore | None" = None,
        scan_id: str | None = None,
    ) -> None:
        self.executor = executor
        self.policy = policy
        self.session_id = session_id
        self.actor = actor
        self.resource_owner_resolver = resource_owner_resolver
        self.trace_store = trace_store
        self.scan_id = scan_id or f"guard_{session_id}"
        self._engine = PolicyEngine(policy)
        self._calls_this_session = 0
        self.decision_log: list[PolicyDecision] = []

    @property
    def tool_definitions(self) -> Any:
        """Delegate tool_definitions to the wrapped executor so Guard is drop-in compatible."""
        return getattr(self.executor, "tool_definitions", [])

    @property
    def authenticated_customer_id(self) -> str:
        return getattr(self.executor, "authenticated_customer_id", self.actor)

    @authenticated_customer_id.setter
    def authenticated_customer_id(self, value: str) -> None:
        self.actor = value
        if hasattr(self.executor, "authenticated_customer_id"):
            self.executor.authenticated_customer_id = value

    def _record_to_trace_store(self, record: ToolExecutionRecord, decision: PolicyDecision) -> None:
        if self.trace_store is None:
            return
        from agentshield.tracing.models import ScanSessionRecord, ToolTraceEvent

        self.trace_store.ensure_session(
            ScanSessionRecord(
                scan_id=self.scan_id,
                target_name="guard_middleware",
                target_type="guard",
                observability="instrumented",
                telemetry_available=True,
            )
        )
        self.trace_store.record_tool_event(
            ToolTraceEvent(
                scan_id=self.scan_id,
                session_id=self.session_id,
                tool_call_id=record.tool_call_id,
                tool_name=record.tool_name,
                arguments=record.arguments,
                result=record.output,
                authorization="allowed" if record.authorized else "denied",
                execution_status="completed" if record.status == "success" else record.status,
                state_changed=record.state_changed,
                latency_ms=record.latency_ms,
                error=None if decision.allowed else decision.deny_detail,
            )
        )

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        approval_token: ApprovalToken | None = None,
    ) -> ToolExecutionRecord:
        """Evaluate policy then execute (or deny) the tool call."""
        resolved_owner = (
            self.resource_owner_resolver(tool_name, arguments)
            if self.resource_owner_resolver is not None
            else None
        )
        ctx = GuardContext(
            actor=self.actor,
            tool_name=tool_name,
            arguments=arguments,
            session_id=self.session_id,
            approval_token=approval_token,
            calls_this_session=self._calls_this_session,
            resolved_owner=resolved_owner,
        )
        decision = self._engine.evaluate(ctx)
        self.decision_log.append(decision)

        if not decision.allowed:
            logger.warning(
                "AgentShieldGuard DENIED | session=%s actor=%s tool=%s reason=%s detail=%s",
                self.session_id,
                self.actor,
                tool_name,
                decision.deny_reason,
                decision.deny_detail,
            )
            denied_record = ToolExecutionRecord(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments,
                authorized=False,
                status="denied",
                state_changed=False,
                output={"error": decision.deny_detail or f"Denied: {decision.deny_reason}"},
            )
            self._record_to_trace_store(denied_record, decision)
            return denied_record

        # Policy approved — forward to underlying executor
        self._calls_this_session += 1
        logger.debug(
            "AgentShieldGuard ALLOWED | session=%s actor=%s tool=%s",
            self.session_id,
            self.actor,
            tool_name,
        )
        record = self.executor.execute(tool_name, arguments, tool_call_id=tool_call_id)
        self._record_to_trace_store(record, decision)
        return record

    @property
    def calls_this_session(self) -> int:
        """Number of tool calls that passed guard and were forwarded to the executor."""
        return self._calls_this_session

    @property
    def denied_count(self) -> int:
        """Number of tool calls denied by the guard in this session."""
        return sum(1 for d in self.decision_log if not d.allowed)

    @property
    def allowed_count(self) -> int:
        """Number of tool calls allowed by the guard in this session."""
        return sum(1 for d in self.decision_log if d.allowed)
