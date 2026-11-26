"""Unit tests for Phase 3 target harnesses and capability discovery."""

from __future__ import annotations

import json

import httpx
import pytest

from agentshield.harnesses import (
    DiscoveredTool,
    HTTPHarness,
    HarnessConfigError,
    LangGraphHarness,
    MCPHarness,
    PythonCallableHarness,
    SessionContext,
    TargetConfig,
    TargetUnauthorizedError,
)
from agentshield.providers import GroqProvider, ProviderConfig
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB
from agentshield.targets.support.http_app import create_support_http_app

pytestmark = pytest.mark.unit


def _auth_config(name: str, **kwargs: object) -> TargetConfig:
    return TargetConfig(target_name=name, authorized=True, **kwargs)


def test_harness_requires_explicit_authorization_and_disables_mock_fallback() -> None:
    with pytest.raises(TargetUnauthorizedError):
        PythonCallableHarness(lambda p: p, TargetConfig(target_name="unauth", authorized=False))

    with pytest.raises(HarnessConfigError):
        PythonCallableHarness(
            lambda p: p,
            TargetConfig(target_name="mock", authorized=True, live_scan=True, allow_mock_fallback=True),
        )


def test_python_black_box_does_not_invent_tools_or_traces() -> None:
    def simple_agent(prompt: str) -> str:
        return f"Echo: {prompt}"

    harness = PythonCallableHarness(simple_agent, _auth_config("py-blackbox"), observability="black_box")
    caps = harness.discover_capabilities()
    assert caps.observability == "black_box"
    assert caps.telemetry_available is False
    assert caps.session_handling == "stateless"
    assert caps.tools == []

    res = harness.execute("Hello")
    assert res.response_text == "Echo: Hello"
    assert res.telemetry_available is False
    assert res.execution_trace is None


def test_python_instrumented_callable_with_session() -> None:
    def instrumented_agent(prompt: str, session: SessionContext) -> dict[str, object]:
        return {
            "response_text": f"Handled for {session.authenticated_subject}",
            "tool_executions": [
                {"tool_name": "lookup", "arguments": {"q": prompt}, "authorized": True, "status": "success"}
            ],
        }

    harness = PythonCallableHarness(
        instrumented_agent,
        _auth_config(
            "py-inst",
            explicit_tools=[DiscoveredTool(name="lookup", description="Lookup tool")],
        ),
        observability="instrumented",
    )
    caps = harness.discover_capabilities()
    assert caps.session_handling == "session_bound"
    assert caps.telemetry_available is True
    assert len(caps.tools) == 1

    res = harness.execute("test", session=SessionContext(authenticated_subject="cust_alice"))
    assert res.telemetry_available is True
    assert res.execution_trace is not None
    assert res.execution_trace[0].tool_name == "lookup"


def test_langgraph_harness_discovers_and_executes_support_agent() -> None:
    idx = 0
    responses = [
        {
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
                                    "name": "get_order",
                                    "arguments": json.dumps({"order_id": "ord_2001"}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        {
            "model": "openai/gpt-oss-20b",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Access denied."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        },
    ]

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal idx
        payload = responses[idx]
        idx += 1
        return httpx.Response(200, json=payload)

    provider = GroqProvider(
        ProviderConfig(api_key="test-key", model="openai/gpt-oss-20b", max_retries=0),
        transport=httpx.MockTransport(handler),
    )

    with SupportSandboxDB(":memory:") as db:
        agent = CustomerSupportAgent(provider, db, variant="hardened", authenticated_customer_id="cust_alice")
        harness = LangGraphHarness(agent, _auth_config("support-lg"))

        caps = harness.discover_capabilities()
        assert caps.observability == "instrumented"
        assert caps.telemetry_available is True
        assert len(caps.tools) == 5

        res = harness.execute("Show me order ord_2001")
        assert res.response_text == "Access denied."
        assert res.execution_trace is not None
        assert res.execution_trace[0].authorized is False
        assert res.execution_trace[0].status == "denied"


def test_http_harness_black_box_vs_instrumented_fastapi() -> None:
    def black_box_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/capabilities":
            return httpx.Response(404)
        return httpx.Response(200, json={"response": "Black-box answer"})

    bb_harness = HTTPHarness(
        "http://blackbox.local",
        _auth_config("http-bb"),
        transport=httpx.MockTransport(black_box_handler),
    )
    bb_caps = bb_harness.discover_capabilities()
    assert bb_caps.observability == "black_box"
    assert bb_caps.telemetry_available is False
    assert bb_caps.tools == []

    bb_res = bb_harness.execute("Hi")
    assert bb_res.response_text == "Black-box answer"
    assert bb_res.execution_trace is None

    # Instrumented FastAPI app via FastAPI TestClient
    from fastapi.testclient import TestClient

    provider = GroqProvider(
        ProviderConfig(api_key="test-key", model="openai/gpt-oss-20b", max_retries=0),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "model": "openai/gpt-oss-20b",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                },
            )
        ),
    )
    with SupportSandboxDB(":memory:") as db:
        agent = CustomerSupportAgent(provider, db, variant="hardened", authenticated_customer_id="cust_alice")
        app = create_support_http_app(agent, expose_telemetry=True)
        tc = TestClient(app)

        def asgi_bridge(request: httpx.Request) -> httpx.Response:
            resp = tc.request(
                request.method,
                request.url.path,
                headers=dict(request.headers),
                content=request.content,
            )
            return httpx.Response(resp.status_code, headers=dict(resp.headers), content=resp.content)

        http_harness = HTTPHarness(
            "http://testserver",
            _auth_config("http-inst"),
            transport=httpx.MockTransport(asgi_bridge),
        )
        inst_caps = http_harness.discover_capabilities()
        assert inst_caps.observability == "instrumented"
        assert len(inst_caps.tools) == 5

        inst_res = http_harness.execute("Hello")
        assert inst_res.response_text == "OK"
        assert inst_res.telemetry_available is True


def test_mcp_harness_discovers_tools_and_executes_tool_call() -> None:
    def mcp_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body["method"]
        if method == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "search_faq",
                                "description": "Search FAQ",
                                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
                            }
                        ]
                    },
                },
            )
        if method == "tools/call":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {
                        "isError": False,
                        "content": [{"type": "text", "text": "Refunds allowed within 30 days."}],
                    },
                },
            )
        return httpx.Response(404)

    mcp = MCPHarness(
        "http://mcp.local/rpc",
        _auth_config("mcp-test"),
        transport=httpx.MockTransport(mcp_handler),
    )
    caps = mcp.discover_capabilities()
    assert caps.target_type == "mcp"
    assert len(caps.tools) == 1
    assert caps.tools[0].source == "mcp_metadata"

    res = mcp.execute(json.dumps({"tool": "search_faq", "arguments": {"query": "refund"}}))
    assert "30 days" in res.response_text
    assert res.execution_trace is not None
    assert res.execution_trace[0].tool_name == "search_faq"
