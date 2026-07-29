"""SSRF guard. socket.getaddrinfo is faked — no real DNS."""

import socket

import pytest

from src.core import net
from src.core.net import UnsafeURLError, assert_safe_url, is_safe_url

# ----- test doubles -----


def _fake_getaddrinfo(ip: str):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET

    def _resolve(host, *_a, **_k):
        return [(family, None, None, "", (ip, 0))]

    return _resolve


# ----- assert_safe_url / is_safe_url -----


@pytest.mark.parametrize(
    ("url", "resolved_ip", "safe"),
    [
        pytest.param("https://example.com/x", "93.184.216.34", True, id="public_https"),
        pytest.param("http://example.com/x", "8.8.8.8", True, id="public_http"),
        pytest.param("http://localhost/x", "127.0.0.1", False, id="loopback"),
        pytest.param("http://10.0.0.1/x", "10.0.0.1", False, id="private_rfc1918_a"),
        pytest.param("http://192.168.1.1/", "192.168.1.1", False, id="private_rfc1918_c"),
        pytest.param("http://metadata/x", "169.254.169.254", False, id="link_local_metadata"),
        pytest.param("http://any/", "0.0.0.0", False, id="unspecified"),
        pytest.param("http://mc/", "224.0.0.1", False, id="multicast"),
        pytest.param("http://res/", "240.0.0.1", False, id="reserved"),
        pytest.param("http://v6/", "::1", False, id="loopback_v6"),
        pytest.param("https://rebind.example/x", "10.1.2.3", False, id="dns_rebind_to_private"),
    ],
)
def test_is_safe_url_by_resolved_ip(monkeypatch, url, resolved_ip, safe):
    monkeypatch.setattr(net.socket, "getaddrinfo", _fake_getaddrinfo(resolved_ip))
    assert is_safe_url(url) is safe


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/x", "file:///etc/passwd", "gopher://example.com/", "no-scheme.com/x"],
)
def test_assert_safe_url_rejects_non_http_scheme(url):
    with pytest.raises(UnsafeURLError, match="scheme"):
        assert_safe_url(url)


def test_assert_safe_url_rejects_missing_host():
    with pytest.raises(UnsafeURLError, match="host"):
        assert_safe_url("http:///path")


def test_assert_safe_url_rejects_unresolvable_host(monkeypatch):
    def _boom(*_a, **_k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(net.socket, "getaddrinfo", _boom)
    with pytest.raises(UnsafeURLError, match="resolve"):
        assert_safe_url("https://nope.invalid/")
