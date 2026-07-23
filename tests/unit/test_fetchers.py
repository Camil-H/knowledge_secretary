"""Deterministic fetcher logic. Network/parse/subprocess collaborators are stubbed."""

import json
import logging
import subprocess
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from src.fetchers import biorxiv, pubmed, rss, x, youtube
from src.fetchers import url as url_fetcher

# ----- test doubles -----


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _BadJsonResp:
    def json(self):
        raise ValueError("not json")


def _raiser(exc):
    def _raise(*_a, **_k):
        raise exc

    return _raise


# ----- youtube.video_id_from_url -----


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://example.com/no-id-here", None),
        ("", None),
    ],
)
def test_video_id_from_url(url, expected):
    assert youtube.video_id_from_url(url) == expected


# ----- x._extract -----


@pytest.mark.parametrize(
    "data,expected_len",
    [
        ([{"id": 1}], 1),  # top-level array
        ({"tweets": [{"id": 1}, {"id": 2}]}, 2),  # wrapped under a known key
        ({"data": [{"id": 1}]}, 1),
    ],
)
def test_x_extract_reads_known_shapes(data, expected_len):
    assert len(x._extract(data)) == expected_len


@pytest.mark.parametrize("data", [{"nope": 5}, "garbage", 42])
def test_x_extract_raises_on_unexpected(data):
    with pytest.raises(x.UnexpectedXFormat):
        x._extract(data)


# ----- x.recent_tweets -----


def test_recent_tweets_composes_argv_and_parses_stdout(monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return type("Proc", (), {"stdout": json.dumps({"tweets": [{"id": 1}, {"id": 2}]})})()

    monkeypatch.setattr(x.subprocess, "run", fake_run)
    out = x.recent_tweets("someuser", limit=5)

    assert captured["argv"] == ["twitter", "user-posts", "someuser", "--max", "5", "--json"]
    assert captured["kwargs"]["check"] is True
    assert out == [{"id": 1}, {"id": 2}]


@pytest.mark.parametrize(
    "fake_run",
    [
        _raiser(subprocess.SubprocessError("cli crashed")),
        lambda *a, **k: type("Proc", (), {"stdout": "not json"})(),  # -> JSONDecodeError
    ],
)
def test_recent_tweets_degrades_on_subprocess_or_json_error(monkeypatch, caplog, fake_run):
    monkeypatch.setattr(x.subprocess, "run", fake_run)
    with caplog.at_level(logging.WARNING):
        out = x.recent_tweets("someuser")
    assert out == []
    assert any("degraded" in r.message for r in caplog.records)


def test_recent_tweets_propagates_unexpected_format(monkeypatch):
    monkeypatch.setattr(
        x.subprocess,
        "run",
        lambda *a, **k: type("Proc", (), {"stdout": json.dumps({"nope": 5})})(),
    )
    with pytest.raises(x.UnexpectedXFormat):
        x.recent_tweets("someuser")


# ----- rss -----


def test_rss_published_utc_from_struct_time():
    st = time.strptime("2024-07-15 12:00:00", "%Y-%m-%d %H:%M:%S")
    assert rss._published_utc({"published_parsed": st}) == datetime(2024, 7, 15, 12, 0, tzinfo=UTC)


def test_rss_published_utc_none_when_missing():
    assert rss._published_utc({}) is None


def test_rss_fetch_normalizes_entries(monkeypatch):
    entry = {
        "id": "e1",
        "title": "T",
        "link": "http://l",
        "summary": "s",
        "published_parsed": time.strptime("2024-01-02", "%Y-%m-%d"),
    }

    class _Parsed:
        feed = {"title": "Feed"}
        entries = [entry]

    monkeypatch.setattr(rss.feedparser, "parse", lambda _u: _Parsed())
    out = rss.fetch("http://x")
    assert out["title"] == "Feed"
    assert len(out["entries"]) == 1
    e = out["entries"][0]
    assert (e["id"], e["title"], e["link"]) == ("e1", "T", "http://l")
    assert e["published"] == datetime(2024, 1, 2, tzinfo=UTC)
    assert e["raw"] is entry


def test_rss_fetch_entry_id_falls_back_to_link(monkeypatch):
    entry = {"link": "http://only-link", "title": "T"}

    class _Parsed:
        feed = {"title": "Feed"}
        entries = [entry]

    monkeypatch.setattr(rss.feedparser, "parse", lambda _u: _Parsed())
    out = rss.fetch("http://x")
    assert out["entries"][0]["id"] == "http://only-link"


def test_rss_fetch_degrades_on_parse_error(monkeypatch):
    monkeypatch.setattr(rss.feedparser, "parse", _raiser(ValueError("malformed feed")))
    assert rss.fetch("http://x") == {"title": "", "entries": []}


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


# ----- pubmed.search_recent -----


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
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return _FakeResp({"esearchresult": {"idlist": []}})

    monkeypatch.setattr(pubmed.httpx, "get", fake_get)
    out = pubmed.search_recent(["q"], datetime.now(UTC) - timedelta(days=1))

    assert out == []
    assert len(calls) == 1  # esummary never called


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


# ----- biorxiv.recent -----


def test_biorxiv_recent_filters_case_insensitively_and_skips_incomplete(monkeypatch):
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
    monkeypatch.setattr(biorxiv.httpx, "get", lambda *a, **k: _FakeResp({"collection": collection}))

    out = biorxiv.recent(["Neuroscience"], datetime(2024, 1, 1, tzinfo=UTC))

    assert [e["doi"] for e in out] == ["10.1/aaa", "10.1/bbb"]
    assert out[0] == {
        "doi": "10.1/aaa",
        "title": "T1",
        "abstract": "A1",
        "published": datetime(2024, 3, 1, tzinfo=UTC),
        "category": "Neuroscience",
    }
    assert out[0]["published"].tzinfo is UTC


@pytest.mark.parametrize(
    "fake_get",
    [
        _raiser(httpx.HTTPError("network down")),
        lambda *a, **k: _BadJsonResp(),
    ],
)
def test_biorxiv_recent_degrades_on_http_or_json_error(monkeypatch, fake_get):
    monkeypatch.setattr(biorxiv.httpx, "get", fake_get)
    out = biorxiv.recent(["neuroscience"], datetime(2024, 1, 1, tzinfo=UTC))
    assert out == []


# ----- url.article_text -----


def test_article_text_none_when_fetch_fails_without_extracting(monkeypatch):
    extract_calls = []
    monkeypatch.setattr(url_fetcher.trafilatura, "fetch_url", lambda _u: None)
    monkeypatch.setattr(
        url_fetcher.trafilatura, "extract", lambda d: extract_calls.append(d) or "should-not-happen"
    )

    assert url_fetcher.article_text("http://x") is None
    assert extract_calls == []


def test_article_text_returns_extracted_text_on_success(monkeypatch):
    monkeypatch.setattr(url_fetcher.trafilatura, "fetch_url", lambda _u: "<html>raw</html>")
    monkeypatch.setattr(url_fetcher.trafilatura, "extract", lambda d: f"extracted:{d}")

    assert url_fetcher.article_text("http://x") == "extracted:<html>raw</html>"


def test_article_text_none_on_any_exception(monkeypatch):
    monkeypatch.setattr(url_fetcher.trafilatura, "fetch_url", _raiser(RuntimeError("boom")))
    assert url_fetcher.article_text("http://x") is None


# ----- youtube.channel_videos (maps rss.fetch output) -----


def test_channel_videos_maps_and_skips_non_videos(monkeypatch):
    feed = {
        "title": "Chan",
        "entries": [
            {
                "id": "i1",
                "title": "V1",
                "link": "http://w",
                "published": datetime(2024, 1, 1, tzinfo=UTC),
                "summary": "d",
                "raw": {"yt_videoid": "vid00000001"},
            },
            {
                "id": "i2",
                "title": "not-a-video",
                "link": "",
                "published": None,
                "summary": "",
                "raw": {},
            },  # no yt_videoid -> skipped
        ],
    }
    monkeypatch.setattr(youtube.rss, "fetch", lambda _url: feed)
    out = youtube.channel_videos("UCabc")
    assert out["channel"] == "Chan"
    assert [v["video_id"] for v in out["videos"]] == ["vid00000001"]


def test_channel_videos_empty_link_falls_back_to_watch_url(monkeypatch):
    feed = {
        "title": "Chan",
        "entries": [
            {
                "id": "i1",
                "title": "V1",
                "link": "",
                "published": None,
                "summary": "",
                "raw": {"yt_videoid": "vid00000001"},
            },
        ],
    }
    monkeypatch.setattr(youtube.rss, "fetch", lambda _url: feed)
    out = youtube.channel_videos("UCabc")
    assert out["videos"][0]["url"] == "https://www.youtube.com/watch?v=vid00000001"


# ----- youtube.transcript -----


def test_transcript_degrades_to_empty_string_on_failure(monkeypatch):
    monkeypatch.setattr(youtube, "_fetch_transcript_text", _raiser(RuntimeError("blocked")))
    assert youtube.transcript("vid1") == ""


# ----- youtube._segment_text -----


@pytest.mark.parametrize(
    "segment,expected",
    [
        ({"text": "dict segment"}, "dict segment"),  # youtube-transcript-api <= 0.6
        (type("Snippet", (), {"text": "object segment"})(), "object segment"),  # >= 1.0
    ],
)
def test_segment_text_handles_both_api_shapes(segment, expected):
    assert youtube._segment_text(segment) == expected
