"""Shared OpenAI-compatible request/response serialization for Groq and OpenRouter."""

from __future__ import annotations

import json
from typing import Any

from agentshield.providers.base import LLMRequest, LLMResponse, TokenUsage, ToolCall
from agentshield.providers.exceptions import ProviderError


def build_openai_chat_payload(request: LLMRequest, model: str) -> dict[str, Any]:
    """Convert an LLMRequest into an OpenAI-compatible chat completions payload."""
    messages_payload: list[dict[str, Any]] = []
    for msg in request.messages:
        item: dict[str, Any] = {"role": msg.role}
        if msg.content is not None:
            item["content"] = msg.content
        if msg.name is not None:
            item["name"] = msg.name
        if msg.tool_call_id is not None:
            item["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            item["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]
        messages_payload.append(item)

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages_payload,
        "temperature": request.temperature,
    }
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens

    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request.tools
        ]

    if request.response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": request.response_schema,
            },
        }

    return payload


def parse_openai_chat_response(
    data: dict[str, Any],
    *,
    provider: str,
    model: str,
) -> LLMResponse:
    """Parse an OpenAI-compatible chat completions response into an LLMResponse."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError(
            f"Provider '{provider}' returned no choices in response.",
            provider=provider,
        )

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    finish_reason = choice.get("finish_reason")

    tool_calls: list[ToolCall] = []
    for raw_tc in message.get("tool_calls") or []:
        func = raw_tc.get("function") or {}
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Provider '{provider}' returned invalid tool call JSON arguments: {exc}",
                    provider=provider,
                ) from exc
        elif isinstance(raw_args, dict):
            parsed_args = raw_args
        else:
            parsed_args = {}

        tool_calls.append(
            ToolCall(
                id=str(raw_tc.get("id", "")),
                name=str(func.get("name", "")),
                arguments=parsed_args,
            )
        )

    raw_usage = data.get("usage") or {}
    usage = TokenUsage(
        prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
        total_tokens=int(raw_usage.get("total_tokens", 0) or 0),
    )

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        provider=provider,
        model=str(data.get("model") or model),
        latency_ms=0.0,
        usage=usage,
        raw_finish_reason=finish_reason,
    )
