"""SSRF guard. socket.getaddrinfo and httpx.Client are faked — no real DNS, no real fetch."""

import socket

import pytest

from src import config
from src.core import url_guard
from src.core.url_guard import (
    UnsafeURLError,
    assert_safe_url,
    fetch_following_safe_redirects,
    is_safe_url,
)

_PUBLIC_IP = "93.184.216.34"
_PRIVATE_IP = "169.254.169.254"
_HOP_URL = "https://pub.example/hop{}"

# ----- test doubles -----


def _fake_getaddrinfo(ip: str):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET

    def _resolve(host, *_a, **_k):
        return [(family, None, None, "", (ip, 0))]

    return _resolve


def _resolve_by_name(host, *_a, **_k):
    """Hosts named *internal* land in link-local space; every other host is public."""
    ip = _PRIVATE_IP if "internal" in host else _PUBLIC_IP
    return [(socket.AF_INET, None, None, "", (ip, 0))]


class _FakeResponse:
    def __init__(self, status_code: int = 200, location: str | None = None, text: str = ""):
        self.status_code = status_code
        self.headers = {"location": location} if location else {}
        self.text = text


class _ScriptedClient:
    """Stands in for httpx.Client: answers each URL from a script and records the GETs."""

    def __init__(self, script: dict[str, _FakeResponse]):
        self._script = script
        self.gets: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def get(self, url):
        self.gets.append(url)
        return self._script[url]


def _patch_transport(monkeypatch, script: dict[str, _FakeResponse]) -> _ScriptedClient:
    client = _ScriptedClient(script)
    monkeypatch.setattr(url_guard.socket, "getaddrinfo", _resolve_by_name)
    monkeypatch.setattr(url_guard.httpx, "Client", lambda **_k: client)
    return client


def _chain(hops: int) -> dict[str, _FakeResponse]:
    """A `hops`-long redirect chain ending in a 200 whose body is "arrived"."""
    script = {
        _HOP_URL.format(i): _FakeResponse(301, location=_HOP_URL.format(i + 1)) for i in range(hops)
    }
    script[_HOP_URL.format(hops)] = _FakeResponse(text="arrived")
    return script


# ----- assert_safe_url / is_safe_url -----


@pytest.mark.parametrize(
    ("url", "resolved_ip", "safe"),
    [
        pytest.param("https://example.com/x", "93.184.216.34", True, id="public_https"),
        pytest.param("http://example.com/x", "8.8.8.8", True, id="public_http"),
        pytest.param("http://localhost/x", "127.0.0.1", False, id="loopback"),
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
    monkeypatch.setattr(url_guard.socket, "getaddrinfo", _fake_getaddrinfo(resolved_ip))
    assert is_safe_url(url) is safe


@pytest.mark.parametrize(
    ("url", "match"),
    [
        pytest.param("ftp://example.com/x", "scheme", id="ftp_scheme"),
        pytest.param("file:///etc/passwd", "scheme", id="file_scheme_no_host"),
        pytest.param("no-scheme.com/x", "scheme", id="no_scheme"),
        pytest.param("http:///path", "host", id="missing_host"),
    ],
)
def test_assert_safe_url_rejects_a_malformed_url(url, match):
    with pytest.raises(UnsafeURLError, match=match):
        assert_safe_url(url)


def test_assert_safe_url_rejects_unresolvable_host(monkeypatch):
    def _boom(*_a, **_k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(url_guard.socket, "getaddrinfo", _boom)
    with pytest.raises(UnsafeURLError, match="resolve"):
        assert_safe_url("https://nope.invalid/")


# ----- fetch_following_safe_redirects -----


def test_fetch_following_safe_redirects_returns_the_end_of_a_safe_chain(monkeypatch):
    final = _FakeResponse(text="<html>body</html>")
    client = _patch_transport(
        monkeypatch,
        {
            "https://pub.example/a": _FakeResponse(301, location="https://pub.example/b"),
            "https://pub.example/b": final,
        },
    )

    assert fetch_following_safe_redirects("https://pub.example/a") is final
    assert client.gets == ["https://pub.example/a", "https://pub.example/b"]


def test_fetch_following_safe_redirects_rejects_a_hop_into_private_space(monkeypatch):
    client = _patch_transport(
        monkeypatch,
        {"https://pub.example/a": _FakeResponse(302, location="http://internal.example/creds")},
    )

    with pytest.raises(UnsafeURLError, match="non-public host"):
        fetch_following_safe_redirects("https://pub.example/a")
    assert client.gets == ["https://pub.example/a"]


def test_fetch_following_safe_redirects_resolves_a_relative_location(monkeypatch):
    client = _patch_transport(
        monkeypatch,
        {
            "https://pub.example/news/a": _FakeResponse(303, location="/news/b"),
            "https://pub.example/news/b": _FakeResponse(),
        },
    )

    fetch_following_safe_redirects("https://pub.example/news/a")
    assert client.gets[-1] == "https://pub.example/news/b"


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_FakeResponse(200, location="https://pub.example/b"), id="ok_with_location"),
        pytest.param(_FakeResponse(302), id="redirect_without_location"),
        pytest.param(_FakeResponse(404), id="error_status"),
    ],
)
def test_fetch_following_safe_redirects_returns_a_non_redirect_as_is(monkeypatch, response):
    client = _patch_transport(monkeypatch, {"https://pub.example/a": response})

    assert fetch_following_safe_redirects("https://pub.example/a") is response
    assert client.gets == ["https://pub.example/a"]


def test_fetch_following_safe_redirects_walks_the_whole_hop_budget(monkeypatch):
    _patch_transport(monkeypatch, _chain(config.MAX_REDIRECT_HOPS))
    assert fetch_following_safe_redirects(_HOP_URL.format(0)).text == "arrived"


def test_fetch_following_safe_redirects_raises_past_the_hop_budget(monkeypatch):
    client = _patch_transport(monkeypatch, _chain(config.MAX_REDIRECT_HOPS + 1))

    with pytest.raises(UnsafeURLError, match="more than"):
        fetch_following_safe_redirects(_HOP_URL.format(0))
    assert len(client.gets) == config.MAX_REDIRECT_HOPS + 1
