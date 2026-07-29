"""Adapter/enricher logic for src/tasks/youtube/adapters.py. yt collaborator is stubbed."""

from datetime import UTC, datetime

import pytest

from src.tasks.youtube import adapters
from src.tasks.youtube.adapters import transcript, yt_channel

_SPEC = {"key": "yt_x", "section": "Pure Science", "channel_id": "UC_test"}
_SINCE = datetime(2020, 1, 1, tzinfo=UTC)
_PUBLISHED = datetime(2024, 3, 1, tzinfo=UTC)


def _video(video_id, *, published=_PUBLISHED, title="A title", url=None, summary="summary text"):
    return {
        "video_id": video_id,
        "title": title,
        "url": url or f"https://youtu.be/{video_id}",
        "published": published,
        "summary": summary,
    }


def _item(*, iid="yt:X", url="https://youtu.be/X", text=""):
    return type(
        "_Item",
        (),
        {"id": iid, "url": url, "text": text, "meta": {}},
    )()


# ----- yt_channel -----


def test_yt_channel_maps_videos_to_items(monkeypatch):
    videos = [_video("A"), _video("B", title="Second", summary="other text")]
    monkeypatch.setattr(
        adapters.yt, "channel_videos", lambda channel_id: {"channel": "ChanX", "videos": videos}
    )

    items = yt_channel(_SPEC, _SINCE, state={})

    assert [i.id for i in items] == ["yt:A", "yt:B"]
    assert [i.meta for i in items] == [{"channel": "ChanX"}, {"channel": "ChanX"}]
    assert items[0].source == "yt_x"
    assert items[0].section == "Pure Science"
    assert items[1].title == "Second"
    assert items[1].text == "other text"


def test_yt_channel_skips_videos_with_no_published_date(monkeypatch):
    videos = [_video("A", published=None), _video("B")]
    monkeypatch.setattr(
        adapters.yt, "channel_videos", lambda channel_id: {"channel": "ChanX", "videos": videos}
    )

    items = yt_channel(_SPEC, _SINCE, state={})

    assert [i.id for i in items] == ["yt:B"]


def test_yt_channel_passes_channel_id_from_spec(monkeypatch):
    seen = {}

    def _fake(channel_id):
        seen["channel_id"] = channel_id
        return {"channel": "ChanX", "videos": []}

    monkeypatch.setattr(adapters.yt, "channel_videos", _fake)

    yt_channel(_SPEC, _SINCE, state={})

    assert seen["channel_id"] == "UC_test"


# ----- transcript enricher -----


@pytest.mark.parametrize(
    "item_id,item_url,url_video_id,expected_arg",
    [
        ("yt:A", "https://youtu.be/A", None, "A"),  # yt-prefixed id: prefix stripped, url unused
        ("other:1", "https://youtu.be/B", "B", "B"),  # non-yt id: falls back to url extraction
    ],
)
def test_transcript_resolves_video_id_and_calls_yt_transcript(
    monkeypatch, item_id, item_url, url_video_id, expected_arg
):
    calls = []
    monkeypatch.setattr(adapters.yt, "transcript", lambda vid: calls.append(vid) or "the text")
    monkeypatch.setattr(adapters.yt, "video_id_from_url", lambda url: url_video_id)

    item = _item(iid=item_id, url=item_url, text="")
    result = transcript(item)

    assert calls == [expected_arg]
    assert result.text == "the text"
    assert result.meta["text_source"] == "transcript"
    assert result is item


@pytest.mark.parametrize(
    "fetched,initial_text,expected_text,expected_source",
    [
        ("real transcript", "desc", "real transcript", "transcript"),
        (
            "",
            "desc",
            "desc",
            "description",
        ),  # CI IP block -> empty transcript keeps RSS description
        ("   ", "desc", "desc", "description"),  # whitespace-only transcript counts as empty
        ("", "", "", "title"),  # neither transcript nor description -> title-only, tagged
    ],
    ids=["transcript", "empty-keeps-desc", "blank-keeps-desc", "no-desc-title-only"],
)
def test_transcript_degrades_to_description_when_transcript_empty(
    monkeypatch, fetched, initial_text, expected_text, expected_source
):
    monkeypatch.setattr(adapters.yt, "transcript", lambda vid: fetched)

    result = transcript(_item(iid="yt:A", text=initial_text))

    assert result.text == expected_text
    assert result.meta["text_source"] == expected_source


def test_transcript_no_resolvable_video_id_leaves_text_unchanged_and_skips_yt_call(monkeypatch):
    called = False

    def _fail(vid):
        nonlocal called
        called = True
        return "should not be used"

    monkeypatch.setattr(adapters.yt, "transcript", _fail)
    monkeypatch.setattr(adapters.yt, "video_id_from_url", lambda url: None)

    item = _item(iid="other:1", url="https://example.com/no-id", text="original text")
    result = transcript(item)

    assert called is False
    assert result.text == "original text"
    assert result.meta["text_source"] == "description"
