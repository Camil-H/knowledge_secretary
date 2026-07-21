from datetime import UTC, datetime

from src.core.models import Context, Item
from src.tasks.newsletter import _clean, run


def _cfg():
    return {
        "window_hours": 24,
        "tasks": {
            "newsletter": {
                "sources": [
                    {"key": "pipeline", "kind": "feed", "section": "Blogs", "url": "http://b"},
                ]
            }
        },
    }


def _item(item_id):
    return Item(
        id=item_id,
        source="pipeline",
        section="Blogs",
        title="T",
        url="http://u",
        published=datetime.now(UTC),
        text="body",
    )


def test_run_synthesizes_and_reports_consumed():
    items = [_item("rss:1"), _item("rss:2")]
    ctx = Context(
        cfg=_cfg(),
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: items,
        call=lambda tier, system, user, max_tokens=None: "# Newsletter\nbody",
        log=lambda m: None,
    )
    result = run(ctx)
    assert result.markdown == "# Newsletter\nbody"
    assert set(result.consumed) == {"rss:1", "rss:2"}


def test_run_empty_gather_skips_llm():
    called = []
    ctx = Context(
        cfg=_cfg(),
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: [],
        call=lambda *a, **k: called.append(1) or "x",
        log=lambda m: None,
    )
    result = run(ctx)
    assert result.markdown == ""
    assert not called  # no LLM spend on an empty day


def test_clean_collapses_and_truncates():
    assert _clean("a\n\n  b\t c") == "a b c"
    assert len(_clean("x" * 5000)) == 2000
