"""URL reachability helper. httpx is faked — no real network."""

import asyncio

from src.tasks.podcast import utils


def test_validate_urls_drops_unreachable(monkeypatch):
    class _Resp:
        def __init__(self, code):
            self.status_code = code

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def head(self, url):
            return _Resp(200 if "ok" in url else 404)

        async def get(self, url):
            return _Resp(404)

    monkeypatch.setattr(utils.httpx, "AsyncClient", lambda *a, **k: _FakeClient())
    urls = asyncio.run(utils.validate_urls(["https://ok.com", "https://bad.com"]))
    assert urls == ["https://ok.com"]
