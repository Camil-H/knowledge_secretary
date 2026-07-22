from datetime import UTC, datetime, timedelta

import pytest

from src.core.models import Context, Item
from src.tasks import youtube as youtube_task
from src.tasks.youtube import run

_TEST_SPEC = {
    "key": "yt_x",
    "kind": "yt_channel",
    "section": "Pure Science",
    "handle": "@x",
    "enrich": ["transcript"],
}


@pytest.fixture(autouse=True)
def _patch_sources(monkeypatch):
    # produce() reads the module-level SOURCES for section ordering
    monkeypatch.setattr(youtube_task, "SOURCES", [_TEST_SPEC])


def _video(vid, *, text="transcript body"):
    return Item(
        id=vid,
        source="yt_x",
        section="Pure Science",
        title=f"Vid {vid}",
        url=f"http://y/{vid}",
        published=datetime.now(UTC) - timedelta(hours=1),
        text=text,
        meta={"channel": "ChanX"},
    )


def _ctx(items, call):
    return Context(
        cfg={},
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: items,
        call=call,
        log=lambda m: None,
    )


def test_run_summarizes_new_videos_and_consumes_all():
    videos = [_video("yt:A"), _video("yt:B")]
    result = run(_ctx(videos, lambda tier, system, user, max_tokens=None: "- b1\n- b2\n- b3"))

    assert set(result.consumed) == {"yt:A", "yt:B"}  # dedup already scoped "new"; consume all
    assert "- Pure Science" in result.markdown
    assert "Vid yt:A" in result.markdown and "Vid yt:B" in result.markdown
    assert "- b1" in result.markdown


def test_run_video_without_transcript_gets_note():
    result = run(_ctx([_video("yt:C", text="")], lambda *a, **k: "unused"))
    assert "(no transcript available)" in result.markdown
    assert result.consumed == ["yt:C"]


def test_run_no_new_videos_blank_markdown():
    result = run(_ctx([], lambda *a, **k: "x"))
    assert result.markdown == ""
    assert result.consumed == []
