import logging
from datetime import UTC, datetime, timedelta

import pytest

from src.core.models import Context, Item
from src.tasks.youtube import task as youtube_task
from src.tasks.youtube.task import (
    PROMPT,
    TRANSCRIPT_CHAR_LIMIT,
    _render,
    _section_order,
    _summarize,
    run,
)

_TEST_SPEC = {
    "key": "yt_x",
    "kind": "yt_channel",
    "section": "Pure Science",
    "channel_id": "UC_test",
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
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: items,
        call=call,
        logger=logging.getLogger("test"),
    )


def test_run_summarizes_new_videos_and_consumes_all():
    videos = [_video("yt:A"), _video("yt:B")]
    result = run(_ctx(videos, lambda system, user, max_tokens=None: "- b1\n- b2\n- b3"))

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


# ----- _summarize -----


def test_summarize_composes_request_with_truncated_transcript():
    seen = {}

    def _call(system, user, max_tokens=None):
        seen["system"] = system
        seen["user"] = user
        return "- b1"

    long_text = "x" * (TRANSCRIPT_CHAR_LIMIT * 2)
    item = _video("yt:A", text=long_text)
    item.title = "Some Title"
    item.meta = {"channel": "Some Channel"}

    _summarize(_ctx([], _call), item)

    assert seen["system"] == PROMPT
    assert "Title: Some Title" in seen["user"]
    assert "Channel: Some Channel" in seen["user"]
    assert f"Transcript:\n{long_text[:TRANSCRIPT_CHAR_LIMIT]}" in seen["user"]
    assert long_text[:TRANSCRIPT_CHAR_LIMIT] != long_text  # sanity: truncation actually bites
    assert seen["user"].count("x") == TRANSCRIPT_CHAR_LIMIT


def test_summarize_filters_blank_lines_from_reply():
    raw = "- b1\n\n   \n- b2\n\t\n- b3"
    result = _summarize(_ctx([], lambda system, user, max_tokens=None: raw), _video("yt:A"))
    assert result == ["- b1", "- b2", "- b3"]


# ----- _section_order -----


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        ([], []),
        ([{"section": "A"}], ["A"]),
        ([{"section": "A"}, {"section": "B"}], ["A", "B"]),
        ([{"section": "A"}, {"section": "A"}, {"section": "B"}], ["A", "B"]),
        (
            [{"section": "B"}, {"section": "A"}, {"section": "B"}, {"section": "A"}],
            ["B", "A"],
        ),
    ],
    ids=["empty", "single", "distinct", "immediate-dup", "interleaved-dup"],
)
def test_section_order_dedups_to_first_appearance(specs, expected):
    assert _section_order(specs) == expected


# ----- _render -----


def test_render_orders_sections_by_config_and_omits_empty_section():
    a1, a2, b1 = _video("yt:A1"), _video("yt:A2"), _video("yt:B1")
    grouped = {
        "Beta": [(b1, ["- b bullet"])],
        "Alpha": [(a1, ["- a1 bullet"]), (a2, ["- a2 bullet"])],
    }

    out = _render(["Alpha", "Empty", "Beta"], grouped)

    assert out.index("- Alpha") < out.index("- Beta")  # config order, not dict/insertion order
    assert "Vid yt:A1" in out and "Vid yt:A2" in out and "Vid yt:B1" in out
    assert "Empty" not in out  # section with no entries omitted entirely, header included
