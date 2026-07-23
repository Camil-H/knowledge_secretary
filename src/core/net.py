# core/net.py
"""SSRF guard for outbound fetches: scheme allow-list + private-host rejection.

A URL is safe only when it is http(s) and every address its host resolves to is
a public unicast address. Callers validate before fetching, and re-validate each
redirect hop, so a public URL cannot redirect the fetcher into internal network
space (loopback, RFC1918, link-local metadata endpoints, and the like).
"""

import ipaddress
import socket
from urllib.parse import urlsplit

type IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

_ALLOWED_SCHEMES = frozenset({"http", "https"})


# == Exceptions ==========================================================


class UnsafeURLError(Exception):
    """A URL failed the SSRF guard (bad scheme, missing/unresolvable, or non-public host)."""


# == Guard ===============================================================


def assert_safe_url(url: str) -> None:
    """Raise UnsafeURLError unless `url` is http(s) and its host resolves only to public IPs."""
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme not allowed: {parts.scheme or '(none)'}")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("missing host")
    for ip in _resolve_ips(host):
        if not _is_public_ip(ip):
            raise UnsafeURLError(f"non-public host {host} resolves to {ip}")


def is_safe_url(url: str) -> bool:
    """True when `url` passes `assert_safe_url`."""
    try:
        assert_safe_url(url)
        return True
    except UnsafeURLError:
        return False


# == Helper Functions ==


def _resolve_ips(host: str) -> set[IPAddress]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise UnsafeURLError(f"cannot resolve host {host}") from e
    return {ipaddress.ip_address(str(info[4][0])) for info in infos}


def _is_public_ip(ip: IPAddress) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )
