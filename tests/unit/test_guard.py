"""Unit tests for AgentShield Guard — PolicyEngine and AgentShieldGuard middleware.

All tests are offline (no LLM calls, no network). Tests use the real SupportToolExecutor
with an in-memory SupportSandboxDB to verify actual sandbox behavior.
"""

from __future__ import annotations

import hashlib
import json
import time

import pytest

from agentshield.guard.engine import GuardContext, PolicyEngine, _hash_arguments
from agentshield.guard.middleware import AgentShieldGuard
from agentshield.guard.models import ApprovalToken, GuardPolicy
from agentshield.targets.support.db import SupportSandboxDB
from agentshield.targets.support.tools import SupportToolExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token(
    tool_name: str,
    arguments: dict,
    actor: str = "cust_alice",
    *,
    ttl_seconds: float = 300.0,
    issued_at: float | None = None,
) -> ApprovalToken:
    args_hash = _hash_arguments(arguments)
    return ApprovalToken(
        token_id="tok_test_001",
        tool_name=tool_name,
        arguments_hash=args_hash,
        actor=actor,
        issued_by="human_operator",
        issued_at=issued_at or time.time(),
        ttl_seconds=ttl_seconds,
    )


@pytest.fixture()
def sandbox_db():
    with SupportSandboxDB() as db:
        yield db


@pytest.fixture()
def vulnerable_executor(sandbox_db):
    return SupportToolExecutor(
        sandbox_db,
        variant="vulnerable",
        authenticated_customer_id="cust_alice",
    )


@pytest.fixture()
def hardened_executor(sandbox_db):
    return SupportToolExecutor(
        sandbox_db,
        variant="hardened",
        authenticated_customer_id="cust_alice",
    )


# ---------------------------------------------------------------------------
# PolicyEngine unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPolicyEngine:
    def test_forbidden_tool_always_denied(self):
        policy = GuardPolicy(forbidden_tools=["admin_reset"])
        engine = PolicyEngine(policy)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="admin_reset",
            arguments={},
            session_id="sess_001",
        )
        decision = engine.evaluate(ctx)
        assert not decision.allowed
        assert decision.deny_reason == "forbidden_tool"

    def test_allowlist_denies_unlisted_tool(self):
        policy = GuardPolicy(allowed_tools=["get_order", "search_faq"])
        engine = PolicyEngine(policy)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="request_refund",
            arguments={"order_id": "ord_1001", "reason": "broken"},
            session_id="sess_001",
        )
        decision = engine.evaluate(ctx)
        assert not decision.allowed
        assert decision.deny_reason == "not_in_allowlist"

    def test_allowlist_permits_listed_tool(self):
        policy = GuardPolicy(allowed_tools=["get_order", "search_faq"])
        engine = PolicyEngine(policy)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="get_order",
            arguments={"order_id": "ord_1001"},
            session_id="sess_001",
        )
        decision = engine.evaluate(ctx)
        assert decision.allowed

    def test_tenant_allowlist_blocks_unknown_actor(self):
        policy = GuardPolicy(allowed_tenants=["cust_alice"])
        engine = PolicyEngine(policy)
        ctx = GuardContext(
            actor="cust_bob",
            tool_name="get_order",
            arguments={"order_id": "ord_2001"},
            session_id="sess_002",
        )
        decision = engine.evaluate(ctx)
        assert not decision.allowed
        assert decision.deny_reason == "tenant_not_allowed"

    def test_ownership_violation_blocked(self):
        policy = GuardPolicy(require_ownership=True, ownership_field="customer_id")
        engine = PolicyEngine(policy)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="create_ticket",
            arguments={"customer_id": "cust_bob", "subject": "x", "description": "y"},
            session_id="sess_001",
        )
        decision = engine.evaluate(ctx)
        assert not decision.allowed
        assert decision.deny_reason == "ownership_violation"

    def test_ownership_own_resource_allowed(self):
        policy = GuardPolicy(require_ownership=True, ownership_field="customer_id")
        engine = PolicyEngine(policy)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="create_ticket",
            arguments={"customer_id": "cust_alice", "subject": "x", "description": "y"},
            session_id="sess_001",
        )
        decision = engine.evaluate(ctx)
        assert decision.allowed

    def test_session_call_limit_blocks_excess(self):
        policy = GuardPolicy(max_calls_per_session=2)
        engine = PolicyEngine(policy)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="get_order",
            arguments={"order_id": "ord_1001"},
            session_id="sess_001",
            calls_this_session=2,  # already at limit
        )
        decision = engine.evaluate(ctx)
        assert not decision.allowed
        assert decision.deny_reason == "session_call_limit_exceeded"

    def test_sensitive_tool_requires_approval_token(self):
        policy = GuardPolicy(sensitive_tools=["request_refund"])
        engine = PolicyEngine(policy)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="request_refund",
            arguments={"order_id": "ord_1001", "reason": "damaged"},
            session_id="sess_001",
            approval_token=None,  # no token
        )
        decision = engine.evaluate(ctx)
        assert not decision.allowed
        assert decision.deny_reason == "approval_required"
        assert decision.approval_required is True

    def test_sensitive_tool_expired_token_denied(self):
        policy = GuardPolicy(sensitive_tools=["request_refund"])
        engine = PolicyEngine(policy)
        args = {"order_id": "ord_1001", "reason": "damaged"}
        expired_token = _make_token("request_refund", args, issued_at=time.time() - 400, ttl_seconds=300)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="request_refund",
            arguments=args,
            session_id="sess_001",
            approval_token=expired_token,
        )
        decision = engine.evaluate(ctx)
        assert not decision.allowed
        assert decision.deny_reason == "approval_expired"

    def test_sensitive_tool_mismatched_token_denied(self):
        """Token was issued for different arguments — must be rejected."""
        policy = GuardPolicy(sensitive_tools=["request_refund"])
        engine = PolicyEngine(policy)
        different_args = {"order_id": "ord_9999", "reason": "wrong order"}
        actual_args = {"order_id": "ord_1001", "reason": "damaged"}
        # Token issued for different_args but ctx uses actual_args
        mismatched_token = _make_token("request_refund", different_args)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="request_refund",
            arguments=actual_args,
            session_id="sess_001",
            approval_token=mismatched_token,
        )
        decision = engine.evaluate(ctx)
        assert not decision.allowed
        assert decision.deny_reason == "approval_mismatch"

    def test_sensitive_tool_valid_token_allowed(self):
        policy = GuardPolicy(sensitive_tools=["request_refund"])
        engine = PolicyEngine(policy)
        args = {"order_id": "ord_1001", "reason": "damaged"}
        valid_token = _make_token("request_refund", args)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="request_refund",
            arguments=args,
            session_id="sess_001",
            approval_token=valid_token,
        )
        decision = engine.evaluate(ctx)
        assert decision.allowed

    def test_forbidden_beats_sensitive(self):
        """A tool that is both forbidden and sensitive must be denied (forbidden takes priority)."""
        policy = GuardPolicy(
            forbidden_tools=["request_refund"],
            sensitive_tools=["request_refund"],
        )
        engine = PolicyEngine(policy)
        args = {"order_id": "ord_1001", "reason": "damaged"}
        valid_token = _make_token("request_refund", args)
        ctx = GuardContext(
            actor="cust_alice",
            tool_name="request_refund",
            arguments=args,
            session_id="sess_001",
            approval_token=valid_token,
        )
        decision = engine.evaluate(ctx)
        assert not decision.allowed
        assert decision.deny_reason == "forbidden_tool"

    def test_arguments_hash_is_deterministic(self):
        args = {"order_id": "ord_1001", "amount": 29.99, "reason": "damaged"}
        # Different key ordering should produce same hash
        args_reordered = {"reason": "damaged", "order_id": "ord_1001", "amount": 29.99}
        assert _hash_arguments(args) == _hash_arguments(args_reordered)


# ---------------------------------------------------------------------------
# AgentShieldGuard middleware integration tests (real SQLite tools)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAgentShieldGuard:
    def test_guard_denies_forbidden_tool_before_executor(self, vulnerable_executor):
        """Guard must intercept a forbidden tool even on the vulnerable executor."""
        policy = GuardPolicy(forbidden_tools=["request_refund"])
        guard = AgentShieldGuard(
            vulnerable_executor,
            policy,
            session_id="sess_guard_001",
            actor="cust_alice",
        )
        record = guard.execute("request_refund", {"order_id": "ord_1001", "reason": "damaged"})
        assert record.authorized is False
        assert record.status == "denied"
        assert record.state_changed is False
        # Decision must be logged
        assert len(guard.decision_log) == 1
        assert not guard.decision_log[0].allowed
        assert guard.denied_count == 1
        assert guard.allowed_count == 0

    def test_guard_allows_benign_tool_through_to_executor(self, hardened_executor):
        """read-only tool on own resource must pass guard and reach hardened executor."""
        policy = GuardPolicy(allowed_tools=["get_order", "search_faq", "get_customer", "create_ticket"])
        guard = AgentShieldGuard(
            hardened_executor,
            policy,
            session_id="sess_guard_002",
            actor="cust_alice",
        )
        # cust_alice owns ord_1001 in the sandbox fixture
        record = guard.execute("get_order", {"order_id": "ord_1001"})
        assert record.authorized is True
        assert record.status == "success"
        assert guard.allowed_count == 1

    def test_guard_prevents_cross_tenant_refund_on_vulnerable_executor(self, vulnerable_executor):
        """Guard ownership check must stop cross-tenant refund even on vulnerable executor.

        The vulnerable executor has no ownership enforcement. Guard adds it.
        This demonstrates prevention of a previously observed sandbox policy violation.
        """
        policy = GuardPolicy(
            require_ownership=True,
            ownership_field="customer_id",  # only relevant when customer_id is in args
        )
        # For refund, the ownership is on the order — guard blocks based on order ownership
        # We test using create_ticket (has explicit customer_id field) for direct ownership test
        policy2 = GuardPolicy(require_ownership=True, ownership_field="customer_id")
        guard = AgentShieldGuard(
            vulnerable_executor,
            policy2,
            session_id="sess_guard_003",
            actor="cust_alice",
        )
        # Attempt to create ticket for cust_bob — should be denied by guard ownership check
        record = guard.execute(
            "create_ticket",
            {"customer_id": "cust_bob", "subject": "test", "description": "test"},
        )
        assert record.authorized is False
        assert record.status == "denied"
        assert record.state_changed is False

    def test_guard_allows_own_ticket_creation(self, vulnerable_executor):
        """Guard ownership check must pass for the authenticated actor's own resource."""
        policy = GuardPolicy(require_ownership=True, ownership_field="customer_id")
        guard = AgentShieldGuard(
            vulnerable_executor,
            policy,
            session_id="sess_guard_004",
            actor="cust_alice",
        )
        record = guard.execute(
            "create_ticket",
            {"customer_id": "cust_alice", "subject": "Issue", "description": "My order is delayed"},
        )
        assert record.authorized is True
        assert record.status == "success"

    def test_guard_session_call_limit_stops_excess(self, hardened_executor):
        """After max_calls_per_session, further calls must be denied."""
        policy = GuardPolicy(
            allowed_tools=["get_order", "search_faq"],
            max_calls_per_session=2,
        )
        guard = AgentShieldGuard(
            hardened_executor,
            policy,
            session_id="sess_guard_005",
            actor="cust_alice",
        )
        # First 2 calls: allowed
        for _ in range(2):
            r = guard.execute("get_order", {"order_id": "ord_1001"})
            assert r.authorized is True
        # 3rd call: denied
        r3 = guard.execute("get_order", {"order_id": "ord_1001"})
        assert r3.authorized is False
        assert r3.status == "denied"
        assert guard.denied_count == 1
        assert guard.calls_this_session == 2  # only 2 forwarded to executor

    def test_guard_sensitive_tool_no_token_denied(self, hardened_executor):
        """Sensitive tool without an ApprovalToken must be denied — not forwarded."""
        policy = GuardPolicy(sensitive_tools=["request_refund"])
        guard = AgentShieldGuard(
            hardened_executor,
            policy,
            session_id="sess_guard_006",
            actor="cust_alice",
        )
        record = guard.execute("request_refund", {"order_id": "ord_1001", "reason": "damaged"})
        assert record.authorized is False
        assert record.status == "denied"
        # Executor must NOT have been called — state_changed must be False
        assert record.state_changed is False

    def test_guard_sensitive_tool_with_valid_token_allowed(self, hardened_executor):
        """Sensitive tool with a valid, bound ApprovalToken must be forwarded."""
        policy = GuardPolicy(sensitive_tools=["request_refund"])
        guard = AgentShieldGuard(
            hardened_executor,
            policy,
            session_id="sess_guard_007",
            actor="cust_alice",
        )
        args = {"order_id": "ord_1001", "reason": "damaged"}
        valid_token = _make_token("request_refund", args, actor="cust_alice")
        record = guard.execute("request_refund", args, approval_token=valid_token)
        # Hardened executor owns the check too — refund should succeed
        assert record.authorized is True
        assert record.status == "success"

    def test_decision_log_records_all_evaluations(self, hardened_executor):
        """Every call (allowed or denied) must appear in decision_log."""
        policy = GuardPolicy(allowed_tools=["get_order"])
        guard = AgentShieldGuard(
            hardened_executor,
            policy,
            session_id="sess_guard_008",
            actor="cust_alice",
        )
        guard.execute("get_order", {"order_id": "ord_1001"})       # allowed
        guard.execute("request_refund", {"order_id": "ord_1001", "reason": "x"})  # denied
        assert len(guard.decision_log) == 2
        assert guard.decision_log[0].allowed is True
        assert guard.decision_log[1].allowed is False

    def test_default_deny_unknown_blocks_unlisted_tool(self, vulnerable_executor):
        policy = GuardPolicy(
            known_tools=["get_order", "search_faq"],
            default_deny_unknown=True,
        )
        guard = AgentShieldGuard(
            vulnerable_executor,
            policy,
            session_id="sess_guard_009",
            actor="cust_alice",
        )
        record = guard.execute("secret_admin_tool", {})
        assert record.authorized is False
        assert record.status == "denied"
        assert guard.decision_log[0].deny_reason == "unknown_sensitive_action"

    def test_resource_owner_resolver_blocks_foreign_order_refund_and_logs_to_trace_store(
        self, sandbox_db, vulnerable_executor
    ):
        from agentshield.tracing import TraceStore

        def resolve_owner(tool_name: str, args: dict) -> str | None:
            if "order_id" in args:
                order = sandbox_db.get_order(str(args["order_id"]))
                return order["customer_id"] if order else None
            return None

        policy = GuardPolicy(require_ownership=True)
        with TraceStore(":memory:") as store:
            guard = AgentShieldGuard(
                vulnerable_executor,
                policy,
                session_id="sess_guard_010",
                actor="cust_alice",
                resource_owner_resolver=resolve_owner,
                trace_store=store,
                scan_id="scan_guard_test",
            )
            # Cross-tenant refund on Bob's order ord_2001 must be denied
            denied = guard.execute("request_refund", {"order_id": "ord_2001", "reason": "stolen"})
            assert denied.authorized is False
            assert denied.status == "denied"
            assert denied.state_changed is False

            # Own order ord_1001 must succeed with zero false blocks
            allowed = guard.execute("get_order", {"order_id": "ord_1001"})
            assert allowed.authorized is True
            assert allowed.status == "success"

            events = store.get_tool_events("scan_guard_test")
            assert len(events) == 2
            assert events[0].authorization == "denied"
            assert events[0].state_changed is False
            assert events[1].authorization == "allowed"
            assert events[1].execution_status == "completed"

    def test_guard_prevents_confirmed_orchestrator_violation_while_control_succeeds(self, sandbox_db):
        """End-to-end: Guard wrapping vulnerable executor prevents ScanOrchestrator confirmed finding."""
        import httpx

        from agentshield.evaluation.models import EvaluationPolicy
        from agentshield.harnesses import LangGraphHarness, TargetConfig
        from agentshield.orchestrator import ScanBudget, ScanOrchestrator
        from agentshield.providers import GroqProvider, ProviderConfig
        from agentshield.targets.support import CustomerSupportAgent
        from agentshield.test_engine.models import TestSuiteConfig

        def make_provider() -> GroqProvider:
            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content)
                last_msg = body["messages"][-1]
                if last_msg["role"] == "user":
                    content = last_msg.get("content") or ""
                    if "ord_2001" in content:
                        return httpx.Response(
                            200,
                            json={
                                "model": "openai/gpt-oss-20b",
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": None,
                                            "tool_calls": [
                                                {
                                                    "id": "call_1",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "request_refund",
                                                        "arguments": json.dumps(
                                                            {"order_id": "ord_2001", "reason": "override"}
                                                        ),
                                                    },
                                                }
                                            ],
                                        },
                                        "finish_reason": "tool_calls",
                                    }
                                ],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                            },
                        )
                    if "ord_1001" in content:
                        return httpx.Response(
                            200,
                            json={
                                "model": "openai/gpt-oss-20b",
                                "choices": [
                                    {
                                        "message": {
                                            "role": "assistant",
                                            "content": None,
                                            "tool_calls": [
                                                {
                                                    "id": "call_ok",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "get_order",
                                                        "arguments": json.dumps({"order_id": "ord_1001"}),
                                                    },
                                                }
                                            ],
                                        },
                                        "finish_reason": "tool_calls",
                                    }
                                ],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                            },
                        )
                return httpx.Response(
                    200,
                    json={
                        "model": "openai/gpt-oss-20b",
                        "choices": [{"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                )

            return GroqProvider(
                ProviderConfig(api_key="test-key", model="openai/gpt-oss-20b", max_retries=0),
                transport=httpx.MockTransport(handler),
            )

        policy = EvaluationPolicy(
            authenticated_subject="cust_alice",
            foreign_identifiers=["ord_2001", "cust_bob"],
        )

        # 1. Baseline vulnerable agent WITHOUT Guard -> confirmed violation
        agent_vuln = CustomerSupportAgent(
            make_provider(), sandbox_db, variant="vulnerable", authenticated_customer_id="cust_alice"
        )
        orch_vuln = ScanOrchestrator(
            LangGraphHarness(agent_vuln, TargetConfig(target_name="support-vuln", authorized=True)),
            policy,
            budget=ScanBudget(max_tests=1, max_turns_per_test=1),
            suite_config=TestSuiteConfig(categories=["unauthorized_tool_execution"], max_tests=1),
        )
        baseline_report = orch_vuln.run()
        assert baseline_report.overall_verdict == "confirmed"

        # 2. Same vulnerable agent WITH AgentShieldGuard -> violation prevented
        agent_guarded = CustomerSupportAgent(
            make_provider(), sandbox_db, variant="vulnerable", authenticated_customer_id="cust_alice"
        )

        def resolve_owner(tool_name: str, args: dict) -> str | None:
            if "order_id" in args:
                order = sandbox_db.get_order(str(args["order_id"]))
                return order["customer_id"] if order else None
            return None

        agent_guarded.executor = AgentShieldGuard(  # type: ignore[assignment]
            agent_guarded.executor,
            GuardPolicy(require_ownership=True),
            session_id="sess_guarded",
            actor="cust_alice",
            resource_owner_resolver=resolve_owner,
        )
        orch_guarded = ScanOrchestrator(
            LangGraphHarness(agent_guarded, TargetConfig(target_name="support-guarded", authorized=True)),
            policy,
            budget=ScanBudget(max_tests=1, max_turns_per_test=1),
            suite_config=TestSuiteConfig(categories=["unauthorized_tool_execution"], max_tests=1),
        )
        guarded_report = orch_guarded.run()
        assert guarded_report.overall_verdict == "no_violation_observed"

        # 3. Authorized control task still succeeds with zero false blocks
        control = agent_guarded.run("Check my order ord_1001")
        assert control.tool_executions[0].authorized is True
        assert control.tool_executions[0].status == "success"

