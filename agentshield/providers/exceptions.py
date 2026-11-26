"""Exceptions for AgentShield LLM providers."""


class ProviderError(Exception):
    """Base exception for all LLM provider failures."""

    def __init__(self, message: str, *, provider: str | None = None, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable


class ProviderConfigError(ProviderError):
    """Raised when a provider is misconfigured (e.g., missing API key)."""


class ZeroCostViolationError(ProviderError):
    """Raised when strict zero-cost mode blocks a non-free model or provider."""


class RateLimitError(ProviderError):
    """Raised when a provider returns HTTP 429 or reports quota exhaustion."""

    def __init__(self, message: str, *, provider: str | None = None, retry_after: float | None = None) -> None:
        super().__init__(message, provider=provider, status_code=429, retryable=True)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request exceeds the configured timeout."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message, provider=provider, retryable=True)


class AllProvidersExhaustedError(ProviderError):
    """Raised when all configured free providers fail or exhaust quotas."""

    def __init__(self, message: str, *, errors: list[tuple[str, str]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []
