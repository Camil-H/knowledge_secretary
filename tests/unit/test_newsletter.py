from datetime import UTC, datetime

from src.core.models import Context, Item
from src.tasks.newsletter.task import (
    ITEM_CHAR_LIMIT,
    ITEM_PROMPT,
    PASSTHROUGH_CHARS,
    SYNTHESIS_PROMPT,
    _clean,
    _synthesis_input,
    run,
)

# ----- test doubles -----


class _Recorder:
    """Records (system, user) per call and replies based on which prompt was used."""

    def __init__(self, item_reply="- item bullet", synth_reply="# Newsletter"):
        self.calls: list[tuple[str, str]] = []
        self._item_reply = item_reply
        self._synth_reply = synth_reply

    def __call__(self, tier, system, user, max_tokens=None):
        self.calls.append((system, user))
        return self._item_reply if system == ITEM_PROMPT else self._synth_reply

    def systems(self):
        return [s for s, _ in self.calls]


def _item(item_id, text, section="Blogs"):
    return Item(
        id=item_id,
        source="pipeline",
        section=section,
        title="T",
        url="http://u",
        published=datetime.now(UTC),
        text=text,
    )


def _ctx(items, call):
    return Context(
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: items,
        call=call,
        log=lambda m: None,
    )


# ----- run: two-stage behavior -----


def test_long_item_is_summarized_short_item_passes_through():
    rec = _Recorder()
    items = [_item("rss:short", "tiny"), _item("rss:long", "x" * (PASSTHROUGH_CHARS + 50))]
    result = run(_ctx(items, rec))

    # exactly one per-item summarize (the long one); the short one never hits the LLM
    assert rec.systems().count(ITEM_PROMPT) == 1
    # a synthesis pass ran, and its input carries the passed-through short text
    assert SYNTHESIS_PROMPT in rec.systems()
    synth_user = next(u for s, u in rec.calls if s == SYNTHESIS_PROMPT)
    assert "tiny" in synth_user
    assert result.markdown == "# Newsletter"
    assert set(result.consumed) == {"rss:short", "rss:long"}


def test_irrelevant_items_excluded_from_synthesis_but_still_consumed():
    rec = _Recorder(item_reply="IRRELEVANT")
    items = [_item("rss:off", "x" * (PASSTHROUGH_CHARS + 50))]
    result = run(_ctx(items, rec))

    assert result.markdown == ""  # nothing relevant -> no newsletter
    assert SYNTHESIS_PROMPT not in rec.systems()  # synthesis skipped
    assert result.consumed == ["rss:off"]  # but marked processed, won't recur


def test_empty_gather_skips_all_llm():
    rec = _Recorder()
    result = run(_ctx([], rec))
    assert result.markdown == ""
    assert rec.calls == []


# ----- helpers -----


def test_clean_collapses_whitespace_and_caps_body():
    assert _clean("a\n\n  b\t c") == "a b c"
    assert len(_clean("x" * (ITEM_CHAR_LIMIT * 2))) == ITEM_CHAR_LIMIT


def test_synthesis_input_groups_by_section():
    a = _item("rss:1", "one", section="News")
    b = _item("rss:2", "two", section="Blogs")
    out = _synthesis_input([(a, "sa"), (b, "sb")])
    assert "## News" in out and "## Blogs" in out
    assert "sa" in out and "sb" in out
