# tests/unit/fetchers/test_pubmed.py
"""Deterministic pubmed fetcher logic. The E-utilities transport is stubbed."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from src.fetchers import pubmed
from tests.unit.fetchers.conftest import _BadJsonResp, _FakeResp, _raiser


@pytest.fixture(autouse=True)
def _guard_allows_every_url(_allow_every_url):
    """pubmed guards every URL before fetching; rejection has its own test module."""
    _allow_every_url(pubmed)


# == pubmed.search_recent =====================================================


@pytest.mark.parametrize(
    "since,expected_reldate",
    [
        (datetime.now(UTC) - timedelta(days=3), 3),
        (datetime.now(UTC) + timedelta(days=5), 1),  # 'since' in the future -> clamped to 1
    ],
)
def test_pubmed_search_recent_composes_esearch_request(monkeypatch, since, expected_reldate):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return _FakeResp({"esearchresult": {"idlist": []}})

    monkeypatch.setattr(pubmed.httpx, "get", fake_get)
    pubmed.search_recent(["foo", "bar"], since, retmax=15)

    esearch_url, params = calls[0]
    assert esearch_url == f"{pubmed._EUTILS}/esearch.fcgi"
    assert params == {
        "db": "pubmed",
        "term": "foo OR bar",
        "datetype": "pdat",
        "reldate": expected_reldate,
        "retmax": 15,
        "sort": "date",
        "retmode": "json",
    }


def test_pubmed_search_recent_empty_idlist_skips_esummary(monkeypatch):
    urls = []

    def fake_get(url, params=None, timeout=None):
        urls.append(url)
        return _FakeResp({"esearchresult": {"idlist": []}})

    monkeypatch.setattr(pubmed.httpx, "get", fake_get)
    out = pubmed.search_recent(["q"], datetime.now(UTC) - timedelta(days=1))

    assert out == []
    assert not any("esummary" in u for u in urls)


def test_pubmed_search_recent_skips_rows_without_title(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        if "esearch" in url:
            return _FakeResp({"esearchresult": {"idlist": ["1", "2"]}})
        return _FakeResp(
            {
                "result": {
                    "uids": ["1", "2"],
                    "1": {"title": "", "pubdate": "2024 Jan 1"},
                    "2": {"title": "Has Title", "pubdate": "2024 Feb 2"},
                }
            }
        )

    monkeypatch.setattr(pubmed.httpx, "get", fake_get)
    out = pubmed.search_recent(["q"], datetime.now(UTC) - timedelta(days=1))

    assert out == [
        {"pmid": "2", "title": "Has Title", "published": datetime(2024, 2, 2, tzinfo=UTC)}
    ]


@pytest.mark.parametrize(
    "fake_get",
    [
        _raiser(httpx.HTTPError("network down")),
        lambda *a, **k: _BadJsonResp(),
    ],
)
def test_pubmed_search_recent_degrades_on_http_or_json_error(monkeypatch, fake_get):
    monkeypatch.setattr(pubmed.httpx, "get", fake_get)
    out = pubmed.search_recent(["q"], datetime.now(UTC) - timedelta(days=1))
    assert out == []


# == Helper Functions =========================================================

# ----- pubmed._parse_date -----

_FALLBACK = datetime(2020, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2024 Jan 15", datetime(2024, 1, 15, tzinfo=UTC)),
        ("2024 Jan", datetime(2024, 1, 1, tzinfo=UTC)),
        ("2024", datetime(2024, 1, 1, tzinfo=UTC)),
        ("2024 Jan 15 (Epub ahead of print)", datetime(2024, 1, 15, tzinfo=UTC)),
        ("not a date", _FALLBACK),
    ],
)
def test_pubmed_parse_date(raw, expected):
    assert pubmed._parse_date(raw, _FALLBACK) == expected
