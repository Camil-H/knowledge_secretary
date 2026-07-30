import logging
import re
from datetime import UTC, datetime, timedelta

import pytest

from src import config
from src.core.models import Context, Item
from src.tasks.youtube import task as youtube_task
from src.tasks.youtube.task import (
    PROMPT,
    _batch_input,
    _parse_blocks,
    _render,
    _section_order,
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


def _ids_in(user):
    return re.findall(r"^\[VIDEO (\S+)\]$", user, re.MULTILINE)


def _blocks(ids):
    return "\n\n".join(f"[VIDEO {vid}]\n- {vid} b1\n- {vid} b2\n- {vid} b3" for vid in ids)


class _Call:
    """Stubbed ctx.call, replying with one well-formed block per video id it was handed."""

    def __init__(self, reply=_blocks):
        self.calls = []
        self._reply = reply

    def __call__(self, system, user, max_tokens=None):
        self.calls.append({"system": system, "user": user})
        return self._reply(_ids_in(user))


def test_run_summarizes_new_videos_and_consumes_all():
    videos = [_video("yt:A"), _video("yt:B")]
    result = run(_ctx(videos, _Call()))

    assert set(result.consumed) == {"yt:A", "yt:B"}  # dedup already scoped "new"; consume all
    assert "- Pure Science" in result.markdown
    assert "Vid yt:A" in result.markdown and "Vid yt:B" in result.markdown
    assert "- yt:A b1" in result.markdown and "- yt:B b1" in result.markdown


def test_run_video_without_transcript_gets_note_without_a_call():
    call = _Call()
    result = run(_ctx([_video("yt:C", text="")], call))

    assert "(no transcript available)" in result.markdown
    assert result.consumed == ["yt:C"]
    assert call.calls == []


# ----- Batching -----


def test_run_groups_videos_into_batches_of_config_size():
    size = config.YOUTUBE_BATCH_SIZE
    videos = [_video(f"yt:{n}") for n in range(size * 2 + 1)]
    call = _Call()

    result = run(_ctx(videos, call))

    assert len(call.calls) == 3
    assert [len(_ids_in(c["user"])) for c in call.calls] == [size, size, 1]
    assert all(c["system"] == PROMPT for c in call.calls)
    for video in videos:
        assert f"- {video.id} b1" in result.markdown


def test_run_missing_block_degrades_only_its_own_video():
    videos = [_video("yt:A"), _video("yt:B"), _video("yt:C")]
    dropped = "yt:B"

    result = run(_ctx(videos, _Call(lambda ids: _blocks([i for i in ids if i != dropped]))))

    assert "- yt:A b1" in result.markdown and "- yt:C b1" in result.markdown
    assert result.markdown.count("(summary unavailable)") == 1
    assert f"- {dropped} b1" not in result.markdown


# ----- _batch_input -----


def test_batch_input_heads_each_video_with_its_id_and_truncates_the_transcript():
    long_text = "x" * (config.YOUTUBE_TRANSCRIPT_CHAR_LIMIT * 2)
    first = _video("yt:A", text=long_text)
    first.title = "Some Title"
    first.meta = {"channel": "Some Channel"}

    user = _batch_input((first, _video("yt:B", text="short body")))

    assert _ids_in(user) == ["yt:A", "yt:B"]
    assert "Title: Some Title" in user
    assert "Channel: Some Channel" in user
    assert f"Transcript:\n{long_text[: config.YOUTUBE_TRANSCRIPT_CHAR_LIMIT]}" in user
    assert long_text[: config.YOUTUBE_TRANSCRIPT_CHAR_LIMIT] != long_text
    assert user.count("x") == config.YOUTUBE_TRANSCRIPT_CHAR_LIMIT


# ----- _parse_blocks -----


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "[VIDEO yt:A]\n- a1\n- a2\n\n[VIDEO yt:B]\n- b1",
            {"yt:A": ["- a1", "- a2"], "yt:B": ["- b1"]},
        ),
        ("**[VIDEO yt:A]**\n- a1\n\n## VIDEO yt:B\n   - b1", {"yt:A": ["- a1"], "yt:B": ["- b1"]}),
        ("[VIDEO yt:A]\n\n  \n- a1\n\t\n- a2", {"yt:A": ["- a1", "- a2"]}),
        ("- a1\n- a2", {}),
        ("[VIDEO yt:A]\n- a1\n[VIDEO yt:Z]\n- z1", {"yt:A": ["- a1"]}),
        ("", {}),
    ],
    ids=[
        "two-blocks",
        "decorated-headers",
        "blank-lines",
        "no-header",
        "unknown-id-after-known",
        "empty-reply",
    ],
)
def test_parse_blocks_keys_bullets_by_requested_id(raw, expected):
    assert _parse_blocks(raw, {"yt:A", "yt:B"}) == expected


# ----- _section_order -----


@pytest.mark.parametrize(
    ("specs", "expected"),
    [
        ([], []),
        ([{"section": "A"}, {"section": "B"}], ["A", "B"]),
        (
            [{"section": "B"}, {"section": "A"}, {"section": "B"}, {"section": "A"}],
            ["B", "A"],
        ),
    ],
    ids=["empty", "distinct", "interleaved-dup"],
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
