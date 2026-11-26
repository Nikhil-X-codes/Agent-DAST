"""Target URL validation and SSRF protection for the AgentShield API."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class TargetValidationError(ValueError):
    """Raised when a target URL or configuration violates SSRF or authorization policy."""


_METADATA_HOSTS = frozenset({"169.254.169.254", "metadata.google.internal", "metadata.azure.internal"})


def validate_target_url(
    url: str,
    *,
    allow_localhost: bool = False,
    allowed_hosts: set[str] | None = None,
) -> str:
    """Validate an HTTP/MCP target URL and block SSRF to metadata or private networks."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise TargetValidationError("Target URL must use http or https scheme.")
    if not parsed.hostname:
        raise TargetValidationError("Target URL must include a valid hostname.")

    host = parsed.hostname.lower()
    if host in _METADATA_HOSTS:
        raise TargetValidationError(f"Target host '{host}' is blocked (cloud metadata endpoint).")

    if allowed_hosts and host in allowed_hosts:
        return url

    is_localhost = host in {"localhost", "127.0.0.1", "::1"}
    if is_localhost:
        if not allow_localhost:
            raise TargetValidationError("Localhost targets are disabled by API SSRF policy.")
        return url

    try:
        ip = ipaddress.ip_address(host)
        _check_ip_allowed(ip, allow_localhost=allow_localhost)
    except ValueError:
        # Hostname rather than literal IP: resolve if possible to block private/link-local targets
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
            for info in infos:
                addr = info[4][0]
                _check_ip_allowed(ipaddress.ip_address(addr), allow_localhost=allow_localhost)
        except socket.gaierror as exc:
            raise TargetValidationError(f"Target hostname '{host}' could not be resolved.") from exc

    return url


def _check_ip_allowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, *, allow_localhost: bool) -> None:
    if ip.is_loopback and allow_localhost:
        return
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        raise TargetValidationError(f"Target IP '{ip}' is blocked by SSRF policy.")
