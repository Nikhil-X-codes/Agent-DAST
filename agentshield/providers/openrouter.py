"""OpenRouter free-models LLM provider adapter."""

from __future__ import annotations

from agentshield.providers._openai_compat import (
    build_openai_chat_payload,
    parse_openai_chat_response,
)
from agentshield.providers.base import BaseLLMProvider, LLMRequest, LLMResponse


class OpenRouterProvider(BaseLLMProvider):
    """Adapter for OpenRouter's free-tier models (strictly `:free` or `openrouter/free`)."""

    provider_name = "openrouter"
    env_api_key = "OPENROUTER_API_KEY"
    default_base_url = "https://openrouter.ai/api/v1"
    default_model = "openrouter/free"
    free_models = frozenset(
        {
            "openrouter/free",
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-3.5-lightning:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        }
    )

    @classmethod
    def is_free_model(cls, model: str) -> bool:
        """Allow `openrouter/free` or any explicitly `:free`-suffixed OpenRouter model."""
        return model == "openrouter/free" or model.endswith(":free")

    def _send_request(self, request: LLMRequest, model: str) -> LLMResponse:
        api_key = self._require_api_key()
        payload = build_openai_chat_payload(request, model)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AgentShield",
            "X-Title": "AgentShield",
        }
        data = self._post_json(
            f"{self.base_url}/chat/completions",
            headers=headers,
            payload=payload,
        )
        return parse_openai_chat_response(data, provider=self.provider_name, model=model)
