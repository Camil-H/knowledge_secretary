# tests/unit/test_runner.py
"""gather() reaches sources/enrichers only through the module-level registry names,
so each is swapped for a local fake registry here -- no real fetcher/enricher runs.
run_source_task() is driven through a faked ctx.gather, so gather()'s own logic is
out of scope for those tests."""

import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from src.core.models import Context, Item
from src.tasks import runner
from src.tasks.runner import LOOKBACK_HOURS, gather, run_source_task


class _FakeRegistry:
    """Minimal name->callable lookup mirroring src.core.registry.Registry.get
    (KeyError on a miss, same as the real registry)."""

    def __init__(self, mapping: dict | None = None):
        self._d = dict(mapping or {})

    def get(self, name: str):
        return self._d[name]


def _item(item_id: str, *, published: datetime | None = None, text: str = "body") -> Item:
    return Item(
        id=item_id,
        source="s",
        section="Sec",
        title="t",
        url="http://u",
        published=published or datetime.now(UTC),
        text=text,
    )


def _spec(key: str, *, kind: str = "rss", enrich: list[str] | None = None) -> dict:
    spec = {"key": key, "kind": kind}
    if enrich is not None:
        spec["enrich"] = enrich
    return spec


def _fetcher(items: list[Item]):
    return lambda spec, since, state: items


def _raiser(exc: Exception):
    def _fetch(spec, since, state):
        raise exc

    return _fetch


# ----- gather: dedup + lookback window -----


def test_gather_filters_out_already_seen_items(monkeypatch):
    since = datetime.now(UTC) - timedelta(hours=1)
    seen = _item("rss:seen")
    new = _item("rss:new")
    monkeypatch.setattr(runner, "sources", _FakeRegistry({"rss": _fetcher([seen, new])}))
    state = {"ids": {"rss:seen": "2026-01-01"}, "kv": {}}

    result = gather([_spec("k")], state, since)

    assert result == [new]


def test_gather_drops_new_item_published_before_since(monkeypatch):
    since = datetime.now(UTC) - timedelta(hours=1)
    stale = _item("rss:stale", published=since - timedelta(hours=1))
    fresh = _item("rss:fresh", published=since + timedelta(hours=1))
    monkeypatch.setattr(runner, "sources", _FakeRegistry({"rss": _fetcher([stale, fresh])}))
    state = {"ids": {}, "kv": {}}

    result = gather([_spec("k")], state, since)

    assert result == [fresh]


# ----- gather: enrichment -----


def test_gather_applies_enrichers_in_spec_order(monkeypatch):
    since = datetime.now(UTC) - timedelta(hours=1)
    item = _item("rss:1", text="a")
    monkeypatch.setattr(runner, "sources", _FakeRegistry({"rss": _fetcher([item])}))
    monkeypatch.setattr(
        runner,
        "enrichers",
        _FakeRegistry(
            {
                "append_z": lambda it: replace(it, text=it.text + "z"),
                "upper": lambda it: replace(it, text=it.text.upper()),
            }
        ),
    )
    state = {"ids": {}, "kv": {}}

    result = gather([_spec("k", enrich=["append_z", "upper"])], state, since)

    assert len(result) == 1
    # non-commutative: spec order append_z->upper gives "AZ"; reversed would give "Az"
    assert result[0].text == "AZ"


# ----- gather: per-source isolation -----


def test_gather_source_crash_is_logged_and_skipped_other_specs_still_contribute(
    monkeypatch, caplog
):
    since = datetime.now(UTC) - timedelta(hours=1)
    good_item = _item("good:1")
    monkeypatch.setattr(
        runner,
        "sources",
        _FakeRegistry(
            {
                "bad": _raiser(RuntimeError("boom")),
                "good": _fetcher([good_item]),
            }
        ),
    )
    state = {"ids": {}, "kv": {}}
    specs = [_spec("bad_source", kind="bad"), _spec("good_source", kind="good")]

    with caplog.at_level(logging.ERROR, logger="src.tasks.runner"):
        result = gather(specs, state, since)

    assert result == [good_item]
    assert "bad_source" in caplog.text


def test_gather_unknown_kind_is_a_registry_keyerror_skipped_without_crash(monkeypatch):
    since = datetime.now(UTC) - timedelta(hours=1)
    monkeypatch.setattr(runner, "sources", _FakeRegistry())  # nothing registered
    state = {"ids": {}, "kv": {}}

    result = gather([_spec("mystery", kind="unknown")], state, since)

    assert result == []


# ----- run_source_task: since window -----


def test_run_source_task_since_is_derived_from_lookback_hours_constant():
    captured = {}

    def _gather(specs, since):
        captured["since"] = since
        return []

    ctx = Context(
        state={"ids": {}, "kv": {}},
        gather=_gather,
        call=lambda *a, **k: "",
        logger=logging.getLogger("test"),
    )
    before = datetime.now(UTC) - timedelta(hours=LOOKBACK_HOURS)
    run_source_task(ctx, [_spec("k")], lambda ctx, items: "unused", "Subject")
    after = datetime.now(UTC) - timedelta(hours=LOOKBACK_HOURS)

    assert before <= captured["since"] <= after


# ----- run_source_task: produce + consume -----


def test_run_source_task_empty_gather_skips_produce():
    calls = {"n": 0}

    def _produce(ctx, items):
        calls["n"] += 1
        return "should not be reached"

    ctx = Context(
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: [],
        call=lambda *a, **k: "",
        logger=logging.getLogger("test"),
    )
    result = run_source_task(ctx, [_spec("k")], _produce, "Subject")

    assert calls["n"] == 0
    assert result.markdown == ""
    assert result.consumed == []


def test_run_source_task_consumes_all_gathered_ids_even_when_produce_returns_empty():
    items = [_item("a:1"), _item("a:2")]
    ctx = Context(
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: items,
        call=lambda *a, **k: "",
        logger=logging.getLogger("test"),
    )

    result = run_source_task(ctx, [_spec("k")], lambda ctx, items: "", "Subject")

    assert result.markdown == ""
    assert set(result.consumed) == {"a:1", "a:2"}
