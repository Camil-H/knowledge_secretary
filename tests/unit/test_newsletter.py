from datetime import UTC, datetime

from src.core.models import Context, Item
from src.tasks.newsletter.task import EDITOR_PROMPT, ITEM_CHAR_LIMIT, _editor_input, run


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


# ----- editor input -----


def test_editor_input_groups_by_section_and_trims_bodies():
    long_item = _item("rss:1", "x" * (ITEM_CHAR_LIMIT * 2), section="News")
    short_item = _item("rss:2", "short body", section="Blogs")
    out = _editor_input([long_item, short_item])

    assert "## News" in out and "## Blogs" in out
    assert "http://u" in out and "short body" in out
    assert out.count("x") == ITEM_CHAR_LIMIT  # long body trimmed to the budget
