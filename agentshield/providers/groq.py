"""Groq free-tier LLM provider adapter."""

from __future__ import annotations

from agentshield.providers._openai_compat import (
    build_openai_chat_payload,
    parse_openai_chat_response,
)
from agentshield.providers.base import BaseLLMProvider, LLMRequest, LLMResponse


class GroqProvider(BaseLLMProvider):
    """Adapter for Groq's free-tier OpenAI-compatible inference API."""

    provider_name = "groq"
    env_api_key = "GROQ_API_KEY"
    default_base_url = "https://api.groq.com/openai/v1"
    default_model = "openai/gpt-oss-20b"
    free_models = frozenset(
        {
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "qwen/qwen3-32b",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        }
    )

    def _send_request(self, request: LLMRequest, model: str) -> LLMResponse:
        api_key = self._require_api_key()
        payload = build_openai_chat_payload(request, model)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = self._post_json(
            f"{self.base_url}/chat/completions",
            headers=headers,
            payload=payload,
        )
        return parse_openai_chat_response(data, provider=self.provider_name, model=model)
