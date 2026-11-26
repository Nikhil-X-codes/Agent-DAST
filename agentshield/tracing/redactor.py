"""Credential and sensitive-value redaction before trace persistence or evaluation."""

from __future__ import annotations

import re
from typing import Any

_SENSITIVE_KEY_SUBSTRINGS = (
    "api_key",
    "apikey",
    "authorization",
    "x-goog-api-key",
    "secret",
    "password",
    "passwd",
    "token",
    "credential",
    "private_key",
)

_KEY_SAFE_ALLOWLIST = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "max_tokens",
        "token_usage",
    }
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
)


def redact_text(text: str) -> str:
    """Redact known API key and bearer token patterns from a string."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def redact_sensitive(value: Any, *, _key_hint: str | None = None) -> Any:
    """Recursively redact sensitive keys and token patterns from nested data structures."""
    if _key_hint is not None:
        lowered = _key_hint.lower()
        if lowered not in _KEY_SAFE_ALLOWLIST and any(s in lowered for s in _SENSITIVE_KEY_SUBSTRINGS):
            return "[REDACTED]"

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {k: redact_sensitive(v, _key_hint=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item) for item in value]
    return value
