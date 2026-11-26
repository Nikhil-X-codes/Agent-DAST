"""Exceptions for AgentShield target harnesses and capability discovery."""


class HarnessError(Exception):
    """Base exception for target harness failures."""


class HarnessConfigError(HarnessError):
    """Raised when a target harness is missing explicit configuration or violates scan policy."""


class TargetUnauthorizedError(HarnessError):
    """Raised when attempting a live scan against a target without explicit authorization."""


class TargetExecutionError(HarnessError):
    """Raised when executing a prompt against a target fails."""
