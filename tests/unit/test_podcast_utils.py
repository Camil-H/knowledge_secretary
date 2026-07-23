"""URL reachability helper. httpx is faked — no real network."""

import asyncio
from typing import cast

import httpx
import pytest

from src.tasks.podcast import utils
from src.tasks.podcast.utils import _url_ok


class _Resp:
    def __init__(self, code, headers=None):
        self.status_code = code
        self.headers = headers or {}


def _respond(result):
    """Turn a parametrize row's `result` into a _Resp, or raise it if it's an exception."""
    if isinstance(result, Exception):
        raise result
    return _Resp(result)


class _FakeAsyncClient:
    """Stub for the httpx.AsyncClient collaborator passed into `_url_ok`."""

    def __init__(self, head_result, get_result=None):
        self._head_result = head_result
        self._get_result = get_result

    async def head(self, url):
        return _respond(self._head_result)

    async def get(self, url):
        return _respond(self._get_result)


def test_validate_urls_drops_unreachable(monkeypatch):
    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def head(self, url):
            return _Resp(200 if "ok" in url else 404)

        async def get(self, url):
            return _Resp(404)

    monkeypatch.setattr(utils, "is_safe_url", lambda _u: True)
    monkeypatch.setattr(utils.httpx, "AsyncClient", lambda *a, **k: _FakeClient())
    urls = asyncio.run(utils.validate_urls(["https://ok.com", "https://bad.com"]))
    assert urls == ["https://ok.com"]


def test_validate_urls_drops_unsafe_url(monkeypatch):
    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def head(self, url):
            raise AssertionError("an unsafe URL must not be fetched")

        get = head

    monkeypatch.setattr(utils, "is_safe_url", lambda _u: False)
    monkeypatch.setattr(utils.httpx, "AsyncClient", lambda *a, **k: _FakeClient())
    assert asyncio.run(utils.validate_urls(["http://169.254.169.254"])) == []


def test_validate_urls_follows_safe_redirect(monkeypatch):
    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def head(self, url):
            if url == "https://start.com":
                return _Resp(301, {"location": "https://dest.com"})
            return _Resp(200)

        async def get(self, url):
            return _Resp(404)

    monkeypatch.setattr(utils, "is_safe_url", lambda _u: True)
    monkeypatch.setattr(utils.httpx, "AsyncClient", lambda *a, **k: _FakeClient())
    assert asyncio.run(utils.validate_urls(["https://start.com"])) == ["https://start.com"]


def test_validate_urls_rejects_redirect_to_unsafe_host(monkeypatch):
    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def head(self, url):
            return _Resp(302, {"location": "http://169.254.169.254/"})

        async def get(self, url):
            return _Resp(404)

    monkeypatch.setattr(utils, "is_safe_url", lambda u: "169.254" not in u)
    monkeypatch.setattr(utils.httpx, "AsyncClient", lambda *a, **k: _FakeClient())
    assert asyncio.run(utils.validate_urls(["https://start.com"])) == []


def test_validate_urls_empty_list_skips_client_construction(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("AsyncClient must not be constructed for an empty URL list")

    monkeypatch.setattr(utils.httpx, "AsyncClient", _fail)
    assert asyncio.run(utils.validate_urls([])) == []


@pytest.mark.parametrize(
    ("head_result", "get_result", "expected"),
    [
        pytest.param(500, 200, True, id="get_fallback_succeeds"),
        pytest.param(httpx.ConnectError("boom"), None, False, id="request_raises"),
    ],
)
def test_url_ok_head_fallback_and_error_branches(monkeypatch, head_result, get_result, expected):
    monkeypatch.setattr(utils, "is_safe_url", lambda _u: True)
    client = _FakeAsyncClient(head_result, get_result)
    assert asyncio.run(_url_ok(cast(httpx.AsyncClient, client), "https://example.com")) == expected
