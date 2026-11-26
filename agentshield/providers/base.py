"""Base LLM provider interface, request/response schemas, and zero-cost enforcement."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agentshield.providers.exceptions import (
    ProviderConfigError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    ZeroCostViolationError,
)


class ToolDefinition(BaseModel):
    """JSON-schema definition of a callable tool exposed to an LLM."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolCall(BaseModel):
    """A structured tool invocation emitted by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single chat message exchanged with an LLM provider."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class TokenUsage(BaseModel):
    """Token accounting for a single inference call or cumulative session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: TokenUsage) -> None:
        """Accumulate token counts from another TokenUsage instance."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


class LLMRequest(BaseModel):
    """Normalized request payload for any AgentShield LLM provider."""

    messages: list[Message]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    tools: list[ToolDefinition] | None = None
    response_schema: dict[str, Any] | None = None


class LLMResponse(BaseModel):
    """Normalized response returned by any AgentShield LLM provider."""

    content: str | None = None
    parsed: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    provider: str
    model: str
    latency_ms: float
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw_finish_reason: str | None = None


class ProviderConfig(BaseModel):
    """Configuration shared across all LLM provider adapters."""

    api_key: str | None = Field(default=None, repr=False)
    model: str
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0)
    zero_cost_mode: bool = True
    base_url: str | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for zero-cost LLM provider adapters."""

    provider_name: ClassVar[str]
    env_api_key: ClassVar[str]
    default_base_url: ClassVar[str]
    default_model: ClassVar[str]
    free_models: ClassVar[frozenset[str]] = frozenset()
    supports_tools: ClassVar[bool] = True
    supports_structured_output: ClassVar[bool] = True

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if config is None:
            load_dotenv(override=False)
            config = ProviderConfig(
                api_key=os.getenv(self.env_api_key),
                model=self.default_model,
            )
        elif config.api_key is None:
            load_dotenv(override=False)
            config = config.model_copy(update={"api_key": os.getenv(self.env_api_key)})

        self.config = config
        self.base_url = (config.base_url or self.default_base_url).rstrip("/")
        self.total_usage = TokenUsage()
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds),
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> BaseLLMProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @classmethod
    def is_free_model(cls, model: str) -> bool:
        """Return True if the given model identifier is allowed in zero-cost mode."""
        return model in cls.free_models

    def validate_zero_cost(self, model: str) -> None:
        """Raise ZeroCostViolationError if zero_cost_mode is enabled and model is not free."""
        if self.config.zero_cost_mode and not self.is_free_model(model):
            raise ZeroCostViolationError(
                f"Model '{model}' is not in the free-tier allowlist for provider '{self.provider_name}' "
                "while strict zero-cost mode is enabled.",
                provider=self.provider_name,
            )

    def _require_api_key(self) -> str:
        """Return the configured API key or raise ProviderConfigError without leaking secrets."""
        if not self.config.api_key:
            raise ProviderConfigError(
                f"Missing API key for provider '{self.provider_name}'. "
                f"Set the {self.env_api_key} environment variable.",
                provider=self.provider_name,
            )
        return self.config.api_key

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute inference with zero-cost validation, bounded retries, and token/latency tracking."""
        model = request.model or self.config.model
        self.validate_zero_cost(model)
        self._require_api_key()

        if request.tools and not self.supports_tools:
            raise ProviderError(
                f"Provider '{self.provider_name}' does not support tool calling.",
                provider=self.provider_name,
            )

        last_error: ProviderError | None = None
        for attempt in range(self.config.max_retries + 1):
            start = time.perf_counter()
            try:
                response = self._send_request(request, model)
                latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
                response = response.model_copy(update={"latency_ms": latency_ms})

                if request.response_schema is not None and response.parsed is None:
                    response = self._parse_structured_content(response)

                self.total_usage.add(response.usage)
                return response
            except ZeroCostViolationError:
                raise
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.config.max_retries:
                    raise
                delay = self._compute_retry_delay(attempt, exc)
                if delay > 0:
                    time.sleep(delay)

        assert last_error is not None
        raise last_error

    def _compute_retry_delay(self, attempt: int, exc: ProviderError) -> float:
        """Compute bounded backoff delay in seconds for retryable errors."""
        if isinstance(exc, RateLimitError) and exc.retry_after is not None:
            return min(max(exc.retry_after, 0.0), 5.0)
        return min(self.config.retry_backoff_seconds * (2**attempt), 5.0)

    def _parse_structured_content(self, response: LLMResponse) -> LLMResponse:
        """Parse JSON text content when a structured response_schema was requested."""
        if not response.content:
            raise ProviderError(
                f"Provider '{self.provider_name}' returned empty content for structured output request.",
                provider=self.provider_name,
            )
        text = response.content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Provider '{self.provider_name}' returned invalid JSON for structured output: {exc}",
                provider=self.provider_name,
            ) from exc
        if not isinstance(parsed, dict):
            raise ProviderError(
                f"Provider '{self.provider_name}' structured output must be a JSON object.",
                provider=self.provider_name,
            )
        return response.model_copy(update={"parsed": parsed})

    def _post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform an HTTP POST and map transport/status failures to ProviderError."""
        try:
            resp = self._client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Request to '{self.provider_name}' timed out after {self.config.timeout_seconds}s.",
                provider=self.provider_name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                f"HTTP transport error for '{self.provider_name}': {type(exc).__name__}",
                provider=self.provider_name,
                retryable=True,
            ) from exc

        if resp.status_code == 429:
            retry_after_raw = resp.headers.get("retry-after")
            retry_after = float(retry_after_raw) if retry_after_raw and retry_after_raw.isdigit() else None
            raise RateLimitError(
                f"Rate limit or free-tier quota exceeded for '{self.provider_name}'.",
                provider=self.provider_name,
                retry_after=retry_after,
            )

        if resp.status_code >= 500:
            raise ProviderError(
                f"Server error {resp.status_code} from '{self.provider_name}'.",
                provider=self.provider_name,
                status_code=resp.status_code,
                retryable=True,
            )

        if resp.status_code >= 400:
            detail = self._safe_error_detail(resp)
            raise ProviderError(
                f"Client error {resp.status_code} from '{self.provider_name}': {detail}",
                provider=self.provider_name,
                status_code=resp.status_code,
                retryable=False,
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(
                f"Invalid JSON response from '{self.provider_name}'.",
                provider=self.provider_name,
            ) from exc
        return data

    @staticmethod
    def _safe_error_detail(resp: httpx.Response) -> str:
        """Extract non-sensitive error detail from an HTTP error response."""
        try:
            body = resp.json()
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict) and "message" in err:
                    return str(err["message"])[:200]
                if "message" in body:
                    return str(body["message"])[:200]
        except ValueError:
            pass
        return resp.reason_phrase or "Bad Request"

    def get_usage_summary(self) -> dict[str, int | str]:
        """Return cumulative token usage tracked by this provider instance."""
        return {
            "provider": self.provider_name,
            "model": self.config.model,
            "prompt_tokens": self.total_usage.prompt_tokens,
            "completion_tokens": self.total_usage.completion_tokens,
            "total_tokens": self.total_usage.total_tokens,
        }

    @abstractmethod
    def _send_request(self, request: LLMRequest, model: str) -> LLMResponse:
        """Send a single normalized request to the underlying provider API."""
