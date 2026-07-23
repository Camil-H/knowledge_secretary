"""Deterministic fetcher logic. Network/parse collaborators are stubbed."""

import time
from datetime import UTC, datetime

import pytest

from src.fetchers import pubmed, rss, x, youtube

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
