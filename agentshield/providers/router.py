"""Free-provider fallback router with strict zero-cost enforcement."""

from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv

from agentshield.providers.base import BaseLLMProvider, LLMRequest, LLMResponse, ProviderConfig, TokenUsage
from agentshield.providers.exceptions import (
    AllProvidersExhaustedError,
    ProviderConfigError,
    ProviderError,
    ZeroCostViolationError,
)
from agentshield.providers.gemini import GeminiProvider
from agentshield.providers.groq import GroqProvider
from agentshield.providers.openrouter import OpenRouterProvider

DEFAULT_PROVIDER_ORDER: tuple[str, ...] = ("groq", "gemini", "openrouter")

_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}


class FreeProviderRouter:
    """Routes LLM requests across free-tier providers with bounded fallback and zero-cost guardrails."""

    def __init__(
        self,
        providers: Sequence[BaseLLMProvider],
        *,
        zero_cost_mode: bool = True,
    ) -> None:
        if not providers:
            raise ProviderConfigError("FreeProviderRouter requires at least one configured provider.")

        if zero_cost_mode:
            for provider in providers:
                if not provider.config.zero_cost_mode:
                    raise ZeroCostViolationError(
                        f"Provider '{provider.provider_name}' has zero_cost_mode=False, "
                        "which is forbidden when router zero_cost_mode=True.",
                        provider=provider.provider_name,
                    )

        self.providers = list(providers)
        self.zero_cost_mode = zero_cost_mode

    @classmethod
    def from_env(
        cls,
        *,
        zero_cost_mode: bool | None = None,
        preferred_order: Sequence[str] | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        load_env: bool = True,
    ) -> FreeProviderRouter:
        """Build a FreeProviderRouter from environment variables for configured free providers."""
        if load_env:
            load_dotenv(override=False)
        if zero_cost_mode is None:
            env_flag = os.getenv("AGENTSHIELD_ZERO_COST", "true").strip().lower()
            zero_cost_mode = env_flag not in {"0", "false", "no", "off"}

        order = list(preferred_order or DEFAULT_PROVIDER_ORDER)
        providers: list[BaseLLMProvider] = []

        for name in order:
            provider_cls = _PROVIDER_REGISTRY.get(name.lower())
            if provider_cls is None:
                raise ProviderConfigError(f"Unknown provider '{name}'. Supported: {', '.join(_PROVIDER_REGISTRY)}")
            api_key = os.getenv(provider_cls.env_api_key)
            if api_key:
                providers.append(
                    provider_cls(
                        ProviderConfig(
                            api_key=api_key,
                            model=provider_cls.default_model,
                            timeout_seconds=timeout_seconds,
                            max_retries=max_retries,
                            zero_cost_mode=zero_cost_mode,
                        )
                    )
                )

        if not providers:
            env_vars = ", ".join(cls_.env_api_key for cls_ in _PROVIDER_REGISTRY.values())
            raise ProviderConfigError(
                f"No free-tier provider API keys found in environment. Set at least one of: {env_vars}."
            )

        return cls(providers, zero_cost_mode=zero_cost_mode)

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Attempt inference across free providers in order; never falls back to paid models."""
        errors: list[tuple[str, str]] = []

        for provider in self.providers:
            try:
                return provider.generate(request)
            except ZeroCostViolationError:
                raise
            except ProviderError as exc:
                errors.append((provider.provider_name, str(exc)))

        summary = "; ".join(f"{name}: {err}" for name, err in errors)
        raise AllProvidersExhaustedError(
            f"All configured free providers failed or exhausted quotas ({summary}).",
            errors=errors,
        )

    def get_usage_summary(self) -> dict[str, object]:
        """Return per-provider and aggregate token usage across the router."""
        total = TokenUsage()
        by_provider: list[dict[str, int | str]] = []
        for provider in self.providers:
            total.add(provider.total_usage)
            by_provider.append(provider.get_usage_summary())
        return {
            "total_prompt_tokens": total.prompt_tokens,
            "total_completion_tokens": total.completion_tokens,
            "total_tokens": total.total_tokens,
            "providers": by_provider,
        }

    def close(self) -> None:
        """Close all underlying provider HTTP clients."""
        for provider in self.providers:
            provider.close()

    def __enter__(self) -> FreeProviderRouter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
