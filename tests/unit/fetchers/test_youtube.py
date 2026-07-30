# tests/unit/fetchers/test_youtube.py
"""Deterministic youtube fetcher logic. rss.fetch and the transcript API are stubbed."""

import logging
from datetime import UTC, datetime

import pytest
from youtube_transcript_api import FetchedTranscript, FetchedTranscriptSnippet

from src.fetchers import youtube
from tests.unit.fetchers.conftest import _raiser

# == Test doubles =============================================================


def _snippet(text: str) -> FetchedTranscriptSnippet:
    return FetchedTranscriptSnippet(text=text, start=0.0, duration=1.0)


def _fetched(texts, language_code="en", is_generated=True) -> FetchedTranscript:
    """The real 1.x value object: an iterable of snippet objects, not dicts."""
    return FetchedTranscript(
        snippets=[_snippet(t) for t in texts],
        video_id="vid1",
        language=language_code.upper(),
        language_code=language_code,
        is_generated=is_generated,
    )


class _FakeTranscript:
    def __init__(self, language_code, texts, is_generated=True):
        self.language_code = language_code
        self._texts = texts
        self._is_generated = is_generated

    def fetch(self):
        return _fetched(self._texts, self.language_code, self._is_generated)


class _Listing:
    """A TranscriptList: iterable of transcripts, plus the two finders."""

    def __init__(self, generated=None, manual=None):
        self.generated = generated
        self.manual = manual
        self.requested_langs = []

    def __iter__(self):
        return iter([t for t in (self.generated, self.manual) if t is not None])

    def find_generated_transcript(self, language_codes):
        self.requested_langs.append(list(language_codes))
        if self.generated is None:
            raise RuntimeError("no generated transcript")
        return self.generated

    def find_transcript(self, language_codes):
        self.requested_langs.append(list(language_codes))
        if self.manual is None:
            raise RuntimeError("no transcript found")
        return self.manual


class _FakeTranscriptApi:
    def __init__(self, listing=None, fetched=None, list_error=None):
        self._listing = listing
        self._fetched = fetched
        self._list_error = list_error
        self.fetch_calls = []

    def list(self, _video_id):
        if self._list_error is not None:
            raise self._list_error
        return self._listing

    def fetch(self, video_id, **_kwargs):
        self.fetch_calls.append(video_id)
        if self._fetched is None:
            raise RuntimeError("no transcript available")
        return self._fetched


def _patch_transcript_api(monkeypatch, api):
    monkeypatch.setattr(youtube, "YouTubeTranscriptApi", lambda: api)
    return api


# == youtube.channel_videos (maps rss.fetch output) ============================


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
            {
                "id": "i3",
                "title": "V3",
                "link": "",  # no link -> the watch url is synthesized from the video id
                "published": None,
                "summary": "",
                "raw": {"yt_videoid": "vid00000002"},
            },
        ],
    }
    monkeypatch.setattr(youtube.rss, "fetch", lambda _url: feed)
    out = youtube.channel_videos("UCabc")
    assert out["channel"] == "Chan"
    assert [v["video_id"] for v in out["videos"]] == ["vid00000001", "vid00000002"]
    assert "not-a-video" not in [v["title"] for v in out["videos"]]
    assert out["videos"][0]["url"] == "http://w"
    assert out["videos"][1]["url"] == "https://www.youtube.com/watch?v=vid00000002"


# == youtube.transcript =======================================================


def test_transcript_returns_the_fetched_text(monkeypatch):
    monkeypatch.setattr(youtube, "_fetch_transcript_text", lambda vid: f"text of {vid}")
    assert youtube.transcript("vid1") == "text of vid1"


def test_transcript_degrades_to_empty_string_on_failure(monkeypatch, caplog):
    monkeypatch.setattr(youtube, "_fetch_transcript_text", _raiser(RuntimeError("blocked")))
    with caplog.at_level(logging.WARNING):
        assert youtube.transcript("vid1") == ""
    assert len([r for r in caplog.records if "degraded" in r.message]) == 1


# == youtube.video_id_from_url ================================================


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://example.com/no-id-here", None),
        (None, None),
    ],
)
def test_video_id_from_url(url, expected):
    assert youtube.video_id_from_url(url) == expected


# == Helper Functions =========================================================

# ----- youtube._fetch_transcript_text -----


def test_fetch_transcript_text_prefers_the_generated_transcript(monkeypatch):
    listing = _Listing(
        generated=_FakeTranscript("en", ["hello", "world"]),
        manual=_FakeTranscript("en", ["manually", "typed"]),
    )
    _patch_transcript_api(monkeypatch, _FakeTranscriptApi(listing=listing))
    assert youtube._fetch_transcript_text("vid1") == "hello world"


def test_fetch_transcript_text_falls_back_to_a_manual_transcript(monkeypatch):
    listing = _Listing(manual=_FakeTranscript("en", ["manually", "typed"], is_generated=False))
    _patch_transcript_api(monkeypatch, _FakeTranscriptApi(listing=listing))
    assert youtube._fetch_transcript_text("vid1") == "manually typed"


@pytest.mark.parametrize("lang", ["fr", "de"])
def test_fetch_transcript_text_resolves_a_video_offering_no_english(monkeypatch, lang):
    """Five of the configured channels are French; fetch() alone defaults to English."""
    listing = _Listing(generated=_FakeTranscript(lang, ["bonjour", "le", "monde"]))
    api = _patch_transcript_api(monkeypatch, _FakeTranscriptApi(listing=listing))
    assert youtube._fetch_transcript_text("vid1") == "bonjour le monde"
    assert listing.requested_langs == [[lang]]
    assert api.fetch_calls == []


def test_fetch_transcript_text_falls_back_to_fetch_when_the_listing_fails(monkeypatch):
    api = _patch_transcript_api(
        monkeypatch,
        _FakeTranscriptApi(
            list_error=RuntimeError("no listing"), fetched=_fetched(["plain", "fetch"])
        ),
    )
    assert youtube._fetch_transcript_text("vid1") == "plain fetch"
    assert api.fetch_calls == ["vid1"]


def test_fetch_transcript_text_raises_when_nothing_resolves(monkeypatch):
    _patch_transcript_api(monkeypatch, _FakeTranscriptApi(list_error=RuntimeError("no listing")))
    with pytest.raises(RuntimeError, match="no transcript"):
        youtube._fetch_transcript_text("vid1")
