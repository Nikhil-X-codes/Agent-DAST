"""Google Gemini free-tier LLM provider adapter."""

from __future__ import annotations

from typing import Any

from agentshield.providers.base import BaseLLMProvider, LLMRequest, LLMResponse, TokenUsage, ToolCall
from agentshield.providers.exceptions import ProviderError


class GeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini's free-tier Generative Language API."""

    provider_name = "gemini"
    env_api_key = "GEMINI_API_KEY"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"
    default_model = "gemini-flash-lite-latest"
    free_models = frozenset(
        {
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-flash-latest",
            "gemini-flash-lite-latest",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        }
    )

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        system_parts: list[dict[str, str]] = []
        contents: list[dict[str, Any]] = []

        for msg in request.messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append({"text": msg.content})
                continue

            if msg.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": msg.name or "tool",
                                    "response": {"result": msg.content or ""},
                                }
                            }
                        ],
                    }
                )
                continue

            parts: list[dict[str, Any]] = []
            if msg.content:
                parts.append({"text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        {
                            "functionCall": {
                                "name": tc.name,
                                "args": tc.arguments,
                            }
                        }
                    )
            gemini_role = "model" if msg.role == "assistant" else "user"
            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        generation_config: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.response_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = request.response_schema

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in request.tools
                    ]
                }
            ]

        return payload

    def _send_request(self, request: LLMRequest, model: str) -> LLMResponse:
        api_key = self._require_api_key()
        payload = self._build_payload(request)
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/models/{model}:generateContent"
        data = self._post_json(url, headers=headers, payload=payload)

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderError(
                f"Provider '{self.provider_name}' returned no candidates in response.",
                provider=self.provider_name,
            )

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []

        for idx, part in enumerate(parts):
            if "text" in part and part["text"] is not None:
                text_chunks.append(str(part["text"]))
            elif "functionCall" in part:
                fc = part["functionCall"] or {}
                tool_calls.append(
                    ToolCall(
                        id=f"gemini_call_{idx}",
                        name=str(fc.get("name", "")),
                        arguments=dict(fc.get("args") or {}),
                    )
                )

        usage_meta = data.get("usageMetadata") or {}
        usage = TokenUsage(
            prompt_tokens=int(usage_meta.get("promptTokenCount", 0) or 0),
            completion_tokens=int(usage_meta.get("candidatesTokenCount", 0) or 0),
            total_tokens=int(usage_meta.get("totalTokenCount", 0) or 0),
        )

        return LLMResponse(
            content="\n".join(text_chunks) if text_chunks else None,
            tool_calls=tool_calls,
            provider=self.provider_name,
            model=model,
            latency_ms=0.0,
            usage=usage,
            raw_finish_reason=candidate.get("finishReason"),
        )
