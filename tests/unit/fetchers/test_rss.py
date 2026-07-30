# tests/unit/fetchers/test_rss.py
"""Deterministic rss fetcher logic. The transport and feedparser are stubbed."""

import time
from datetime import UTC, datetime

import httpx
import pytest

from src.fetchers import rss
from tests.unit.fetchers.conftest import _FakeHttpResp, _raiser


@pytest.fixture(autouse=True)
def _guard_allows_every_url(_allow_every_url):
    """rss guards every URL before fetching; rejection has its own test module."""
    _allow_every_url(rss)


# == rss.fetch ================================================================


def test_rss_fetch_normalizes_entries(monkeypatch):
    """One entry of each date/id shape the normalizer has to handle."""
    full = {
        "id": "e1",
        "title": "T",
        "link": "http://l",
        "summary": "s",
        "published_parsed": time.strptime("2024-01-02", "%Y-%m-%d"),
    }
    link_only = {"link": "http://only-link", "title": "T2"}
    updated_only = {
        "id": "e3",
        "title": "T3",
        "link": "http://l3",
        "updated_parsed": time.strptime("2024-07-15 12:00:00", "%Y-%m-%d %H:%M:%S"),
    }

    class _Parsed:
        feed = {"title": "Feed"}
        entries = [full, link_only, updated_only]

    monkeypatch.setattr(rss.httpx, "get", lambda *a, **k: _FakeHttpResp())
    monkeypatch.setattr(rss.feedparser, "parse", lambda _content: _Parsed())
    out = rss.fetch("http://x")

    assert out["title"] == "Feed"
    assert len(out["entries"]) == len(_Parsed.entries)
    normalized, from_link, from_updated = out["entries"]
    assert (normalized["id"], normalized["title"], normalized["link"]) == ("e1", "T", "http://l")
    assert normalized["published"] == datetime(2024, 1, 2, tzinfo=UTC)
    assert normalized["raw"] is full
    assert from_link["id"] == link_only["link"]
    assert from_link["published"] is None
    assert from_updated["published"] == datetime(2024, 7, 15, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("fake_get", "fake_parse"),
    [
        pytest.param(
            lambda *a, **k: _FakeHttpResp(),
            _raiser(ValueError("malformed feed")),
            id="parse_error",
        ),
        pytest.param(
            _raiser(httpx.TimeoutException("timed out")),
            lambda _content: None,
            id="transport_error",
        ),
    ],
)
def test_rss_fetch_degrades_on_a_failed_fetch_or_parse(monkeypatch, fake_get, fake_parse):
    monkeypatch.setattr(rss.httpx, "get", fake_get)
    monkeypatch.setattr(rss.feedparser, "parse", fake_parse)
    assert rss.fetch("http://x") == {"title": "", "entries": []}
