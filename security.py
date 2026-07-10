"""Small, dependency-free security helpers shared by API and project storage."""
from __future__ import annotations

import ipaddress
import os
import socket
from pathlib import Path
from urllib.parse import urlsplit


class SecurityError(ValueError):
    pass


def resolve_within(path: str | Path, root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise SecurityError("path escapes the allowed directory") from exc
    return resolved


def validate_remote_url(url: str, *, allow_private: bool | None = None) -> str:
    value = str(url or "").strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SecurityError("remote URL must use http or https")
    if parsed.username or parsed.password:
        raise SecurityError("remote URL must not contain credentials")
    host = parsed.hostname.lower().rstrip(".")
    blocked_names = {"metadata", "metadata.google.internal", "instance-data"}
    if host in blocked_names:
        raise SecurityError("remote host is not allowed")
    if allow_private is None:
        allow_private = os.getenv("AUDIOBOOKSTUDIO_ALLOW_PRIVATE_REMOTE_URLS", "").lower() in {"1", "true", "yes"}
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, parsed.port or 80, type=socket.SOCK_STREAM)}
        except (OSError, ValueError):
            addresses = set()
    for address in addresses:
        if address.is_link_local or address.is_unspecified or address.is_reserved:
            raise SecurityError("remote host is not allowed")
        if address.is_private and not (allow_private or address.is_loopback):
            raise SecurityError("private remote hosts require AUDIOBOOKSTUDIO_ALLOW_PRIVATE_REMOTE_URLS=1")
    return value
