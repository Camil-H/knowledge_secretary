import logging
from datetime import UTC, datetime

import pytest

from src.core.models import Context, Item
from src.tasks.newsletter.task import (
    EDITOR_PROMPT,
    ITEM_CHAR_FLOOR,
    ITEM_CHAR_LIMIT,
    SYNTHESIS_PROMPT,
    TOTAL_CHAR_BUDGET,
    _editor_input,
    _per_item_budget,
    _produce,
    run,
)


def _item(item_id, text, section="Blogs", title="T"):
    return Item(
        id=item_id,
        source="pipeline",
        section=section,
        title=title,
        url="http://u",
        published=datetime.now(UTC),
        text=text,
    )


def _ctx(items, call):
    return Context(
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: items,
        call=call,
        logger=logging.getLogger("test"),
    )


class _Recorder:
    """Fake ctx.call that records every (system, user) and returns a distinct string."""

    def __init__(self):
        self.calls: list[dict[str, str]] = []

    def __call__(self, system, user, max_tokens=None):
        self.calls.append({"system": system, "user": user})
        return f"# out {len(self.calls)}"


# ----- run: single editor pass -----


def test_run_synthesizes_from_all_items():
    seen = {}

    def _call(system, user, max_tokens=None):
        seen["system"] = system
        seen["user"] = user
        return "# Newsletter"

    items = [_item("rss:1", "Alpha body", section="News"), _item("rss:2", "Beta body")]
    result = run(_ctx(items, _call))

    assert result.markdown == "# Newsletter"
    assert seen["system"] == EDITOR_PROMPT  # one editor pass, not per-item
    assert "Alpha body" in seen["user"] and "Beta body" in seen["user"]
    assert set(result.consumed) == {"rss:1", "rss:2"}


def test_run_empty_gather_skips_the_editor_call():
    calls = {"n": 0}

    def _call(system, user, max_tokens=None):
        calls["n"] += 1
        return "x"

    result = run(_ctx([], _call))
    assert result.markdown == ""
    assert calls["n"] == 0


# ----- produce: adaptive budget (all sources kept) -----


def test_produce_quiet_day_one_call_with_full_items():
    rec = _Recorder()
    # Few enough that each item's share is capped at the full per-item limit.
    n = TOTAL_CHAR_BUDGET // ITEM_CHAR_LIMIT
    items = [_item(f"rss:{i}", "x" * (ITEM_CHAR_LIMIT * 2), title=f"T{i}") for i in range(n)]

    out = _produce(_ctx(items, rec), items)

    assert out == "# out 1"
    assert len(rec.calls) == 1
    assert rec.calls[0]["system"] == EDITOR_PROMPT
    assert _per_item_budget(n) == ITEM_CHAR_LIMIT
    assert rec.calls[0]["user"].count("x") == ITEM_CHAR_LIMIT * n  # full budget per item


def test_produce_busy_day_one_call_trims_but_keeps_all_sources():
    rec = _Recorder()
    # Enough items to force a per-item share below the cap, but still one prompt.
    n = TOTAL_CHAR_BUDGET // ITEM_CHAR_LIMIT + 1
    per_item = _per_item_budget(n)
    assert ITEM_CHAR_FLOOR < per_item < ITEM_CHAR_LIMIT  # genuinely trimmed
    items = [_item(f"rss:{i}", "x" * (ITEM_CHAR_LIMIT * 2), title=f"T{i}") for i in range(n)]

    _produce(_ctx(items, rec), items)

    assert len(rec.calls) == 1
    user = rec.calls[0]["user"]
    assert user.count("x") == per_item * n  # every body trimmed to its share
    for i in range(n):
        assert f"### T{i}" in user  # all sources present


def test_produce_extreme_volume_uses_map_reduce():
    rec = _Recorder()
    # Past the point where even the floor fits one prompt -> map-reduce.
    n = TOTAL_CHAR_BUDGET // ITEM_CHAR_FLOOR + 1
    items = [_item(f"rss:{i}", f"body-{i}", title=f"T{i}") for i in range(n)]

    out = _produce(_ctx(items, rec), items)

    assert len(rec.calls) > 1  # batches + final synthesis
    map_calls = rec.calls[:-1]
    reduce_call = rec.calls[-1]
    assert all(c["system"] == EDITOR_PROMPT for c in map_calls)
    assert reduce_call["system"] == SYNTHESIS_PROMPT
    assert out == f"# out {len(rec.calls)}"  # returns the final synthesis
    # every source is represented across the batch (map) inputs
    mapped = "\n".join(c["user"] for c in map_calls)
    for i in range(n):
        assert f"body-{i}" in mapped


# ----- editor input -----


def test_editor_input_groups_by_section_and_trims_bodies():
    long_item = _item("rss:1", "x" * (ITEM_CHAR_LIMIT * 2), section="News")
    short_item = _item("rss:2", "short body", section="Blogs")
    out = _editor_input([long_item, short_item], ITEM_CHAR_LIMIT)

    assert "## News" in out and "## Blogs" in out
    assert "http://u" in out and "short body" in out
    assert out.count("x") == ITEM_CHAR_LIMIT  # long body trimmed to the budget


@pytest.mark.parametrize("text", ["", "   ", "\t\n  \n"], ids=["empty", "spaces", "whitespace"])
def test_editor_input_blank_text_renders_placeholder(text):
    out = _editor_input([_item("rss:1", text)], ITEM_CHAR_LIMIT)

    assert "(no content available)" in out
    assert out.split("\n")[-1] == "(no content available)"  # not an empty body line
