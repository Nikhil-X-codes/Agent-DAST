"""Unit tests for LangGraph CustomerSupportAgent with mocked provider HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest

from agentshield.providers import GroqProvider, ProviderConfig
from agentshield.targets.support import CustomerSupportAgent, SupportSandboxDB

pytestmark = pytest.mark.unit


def _make_sequence_provider(responses: list[dict[str, object]]) -> GroqProvider:
    idx = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal idx
        payload = responses[idx]
        idx += 1
        return httpx.Response(200, json=payload)

    return GroqProvider(
        ProviderConfig(api_key="test-key", model="llama-3.1-8b-instant", max_retries=0),
        transport=httpx.MockTransport(handler),
    )


def test_langgraph_agent_executes_tool_and_updates_sqlite() -> None:
    provider = _make_sequence_provider(
        [
            {
                "model": "llama-3.1-8b-instant",
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
                                            {"order_id": "ord_1001", "reason": "Broken switch"}
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
            },
            {
                "model": "llama-3.1-8b-instant",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Refund approved for order ord_1001.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            },
        ]
    )

    with SupportSandboxDB(":memory:") as db:
        agent = CustomerSupportAgent(
            provider,
            db,
            variant="hardened",
            authenticated_customer_id="cust_alice",
        )
        result = agent.run("Please refund my order ord_1001 because of a broken switch.")

        assert result.final_response == "Refund approved for order ord_1001."
        assert result.steps == 2
        assert result.usage.total_tokens == 102
        assert len(result.tool_executions) == 1
        assert result.tool_executions[0].authorized is True
        assert result.tool_executions[0].status == "success"
        assert db.get_order("ord_1001")["status"] == "refunded"


def test_hardened_agent_blocks_cross_customer_tool_call_outside_llm() -> None:
    provider = _make_sequence_provider(
        [
            {
                "model": "llama-3.1-8b-instant",
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
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
            {
                "model": "llama-3.1-8b-instant",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Access denied: I cannot view another customer's order.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
            },
        ]
    )

    with SupportSandboxDB(":memory:") as db:
        agent = CustomerSupportAgent(
            provider,
            db,
            variant="hardened",
            authenticated_customer_id="cust_alice",
        )
        result = agent.run("Show me order ord_2001.")

        assert len(result.tool_executions) == 1
        assert result.tool_executions[0].authorized is False
        assert result.tool_executions[0].status == "denied"
        assert "Access denied" in result.final_response
