# tests/unit/fetchers/test_openrxiv.py
"""Deterministic openRxiv fetcher logic. The Details-API transport is stubbed."""

from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx
import pytest

from src.fetchers import openrxiv
from tests.unit.fetchers.conftest import _BadJsonResp, _FakeResp, _raiser


@pytest.fixture(autouse=True)
def _guard_allows_every_url(_allow_every_url):
    """openrxiv guards the host before fetching; rejection has its own test module."""
    _allow_every_url(openrxiv)


# == openrxiv.recent ==========================================================


def test_openrxiv_recent_filters_case_insensitively_and_skips_incomplete(monkeypatch):
    since = datetime(2024, 1, 1, tzinfo=UTC)
    collection = [
        {
            "category": "Neuroscience",
            "doi": "10.1/aaa",
            "title": "T1",
            "abstract": "A1",
            "date": "2024-03-01",
        },
        {
            "category": "neuroscience",  # lowercase -> still matches "Neuroscience" filter
            "doi": "10.1/bbb",
            "title": "T2",
            "abstract": "A2",
            "date": "2024-03-02",
        },
        {
            "category": "Neuroscience",
            "doi": "10.1/bad",
            "title": "malformed date",
            "abstract": "A3",
            "date": "not-a-date",
        },
        {
            "category": "Neuroscience",
            "title": "missing doi",
            "date": "2024-03-03",
        },  # no doi -> skipped
        {
            "category": "Neuroscience",
            "doi": "10.1/ccc",
            "title": "missing date",
        },  # no date -> skipped
        {
            "category": "Genetics",
            "doi": "10.1/ddd",
            "title": "wrong category",
            "date": "2024-03-04",
        },  # filtered out
    ]
    monkeypatch.setattr(
        openrxiv.httpx, "get", lambda *a, **k: _FakeResp({"collection": collection})
    )

    out = openrxiv.recent("biorxiv", ["Neuroscience"], since)

    assert [e["doi"] for e in out] == ["10.1/aaa", "10.1/bbb", "10.1/bad"]
    assert out[0] == {
        "doi": "10.1/aaa",
        "title": "T1",
        "abstract": "A1",
        "published": datetime(2024, 3, 1, tzinfo=UTC),
        "category": "Neuroscience",
    }
    assert out[0]["published"].tzinfo is UTC
    assert out[2]["published"] == since  # an unparseable date falls back, it doesn't drop the batch


@pytest.mark.parametrize(
    "fake_get",
    [
        _raiser(httpx.HTTPError("network down")),
        lambda *a, **k: _BadJsonResp(),
    ],
)
def test_openrxiv_recent_degrades_on_http_or_json_error(monkeypatch, fake_get):
    monkeypatch.setattr(openrxiv.httpx, "get", fake_get)
    out = openrxiv.recent("biorxiv", ["neuroscience"], datetime(2024, 1, 1, tzinfo=UTC))
    assert out == []


def test_openrxiv_recent_asserts_the_host_once_however_many_pages_it_walks(monkeypatch):
    pages = 4
    guarded: list[str] = []
    requested: list[str] = []
    monkeypatch.setattr(openrxiv, "assert_safe_url", lambda url: guarded.append(url))

    def _get(url, **_kwargs):
        requested.append(url)
        entry = {
            "category": "Neuroscience",
            "doi": f"10.1/p{len(requested)}",
            "title": "T",
            "abstract": "A",
            "date": "2024-03-01",
        }
        return _FakeResp({"collection": [entry], "messages": [{"total": pages}]})

    monkeypatch.setattr(openrxiv.httpx, "get", _get)

    out = openrxiv.recent("biorxiv", ["Neuroscience"], datetime(2024, 1, 1, tzinfo=UTC))

    assert len(out) == pages
    assert len(requested) == pages
    assert len(guarded) == 1
    assert {urlsplit(u).netloc for u in requested} == {urlsplit(guarded[0]).netloc}
