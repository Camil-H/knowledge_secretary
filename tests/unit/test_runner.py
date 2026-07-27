# tests/unit/test_runner.py
"""gather() reaches sources/enrichers only through the module-level registry names,
so each is swapped for a local fake registry here -- no real fetcher/enricher runs.
run_source_task() is driven through a faked ctx.gather, so gather()'s own logic is
out of scope for those tests."""

import logging
import re
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from src.core.errors import AuthError
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


# ----- gather: per-source log breakdown -----


@pytest.mark.parametrize(
    "fetched_ids, seen_ids, stale_ids, expected",
    [
        ([], [], [], "0 new of 0 fetched (0 seen, 0 outside window)"),
        (["a", "b"], [], [], "2 new of 2 fetched (0 seen, 0 outside window)"),
        (["a", "b"], ["a", "b"], [], "0 new of 2 fetched (2 seen, 0 outside window)"),
        (["a", "b"], [], ["a", "b"], "0 new of 2 fetched (0 seen, 2 outside window)"),
        (["a", "b", "c"], ["a"], ["b"], "1 new of 3 fetched (1 seen, 1 outside window)"),
        # an item both seen and stale is counted once, so the parts always sum to fetched
        (["a"], ["a"], ["a"], "0 new of 1 fetched (1 seen, 0 outside window)"),
    ],
    ids=["empty-source", "all-new", "all-seen", "all-stale", "mixed", "seen-and-stale"],
)
def test_gather_logs_why_a_source_yielded_what_it_did(
    monkeypatch, caplog, fetched_ids, seen_ids, stale_ids, expected
):
    since = datetime.now(UTC) - timedelta(hours=1)
    items = [
        _item(
            item_id,
            published=(since - timedelta(hours=1))
            if item_id in stale_ids
            else (since + timedelta(hours=1)),
        )
        for item_id in fetched_ids
    ]
    monkeypatch.setattr(runner, "sources", _FakeRegistry({"rss": _fetcher(items)}))
    state = {"ids": dict.fromkeys(seen_ids, "2026-01-01"), "kv": {}}

    with caplog.at_level(logging.INFO, logger="src.tasks.runner"):
        gather([_spec("src_key")], state, since)

    assert f"gather: src_key → {expected}" in caplog.text


def test_gather_log_parts_sum_to_fetched(monkeypatch, caplog):
    # the breakdown is only diagnostic if every fetched item lands in exactly one bucket
    since = datetime.now(UTC) - timedelta(hours=1)
    items = [
        _item("new:1", published=since + timedelta(hours=1)),
        _item("seen:1", published=since + timedelta(hours=1)),
        _item("stale:1", published=since - timedelta(hours=1)),
        _item("stale:2", published=since - timedelta(hours=1)),
    ]
    monkeypatch.setattr(runner, "sources", _FakeRegistry({"rss": _fetcher(items)}))
    state = {"ids": {"seen:1": "2026-01-01"}, "kv": {}}

    with caplog.at_level(logging.INFO, logger="src.tasks.runner"):
        gather([_spec("k")], state, since)

    match = re.search(r"(\d+) new of (\d+) fetched \((\d+) seen, (\d+) outside", caplog.text)
    assert match, f"log line did not match the expected shape: {caplog.text}"
    kept, fetched, seen, stale = (int(n) for n in match.groups())
    assert fetched == len(items)
    assert kept + seen + stale == fetched


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


def test_gather_auth_error_records_notice_and_continues(monkeypatch, caplog):
    since = datetime.now(UTC) - timedelta(hours=1)
    good_item = _item("good:1")
    monkeypatch.setattr(
        runner,
        "sources",
        _FakeRegistry(
            {
                "bad": _raiser(AuthError("x", detail="creds expired")),
                "good": _fetcher([good_item]),
            }
        ),
    )
    state = {"ids": {}, "kv": {}}
    specs = [_spec("x_src", kind="bad"), _spec("good_src", kind="good")]

    with caplog.at_level(logging.ERROR, logger="src.tasks.runner"):
        result = gather(specs, state, since)

    assert result == [good_item]  # auth failure of one source doesn't sink the others
    assert state["_notices"] == ["x_src: creds expired"]
    assert any("auth failed" in r.message for r in caplog.records)


def test_gather_unknown_kind_is_a_registry_keyerror_skipped_without_crash(monkeypatch):
    since = datetime.now(UTC) - timedelta(hours=1)
    monkeypatch.setattr(runner, "sources", _FakeRegistry())  # nothing registered
    state = {"ids": {}, "kv": {}}

    result = gather([_spec("mystery", kind="unknown")], state, since)

    assert result == []


# ----- gather: concurrent fetch -----


def _barrier_fetcher(barrier: threading.Barrier, items: list[Item]):
    """Fetcher that returns only once a second fetch reaches the barrier concurrently;
    run sequentially the first .wait() would time out and break the barrier."""

    def _fetch(spec, since, state):
        barrier.wait()
        return items

    return _fetch


def test_gather_fetches_sources_concurrently_preserving_spec_order(monkeypatch):
    since = datetime.now(UTC) - timedelta(hours=1)
    a, b = _item("a:1"), _item("b:1")
    barrier = threading.Barrier(2, timeout=5)
    monkeypatch.setattr(
        runner,
        "sources",
        _FakeRegistry({"a": _barrier_fetcher(barrier, [a]), "b": _barrier_fetcher(barrier, [b])}),
    )
    state = {"ids": {}, "kv": {}}
    specs = [_spec("a_src", kind="a"), _spec("b_src", kind="b")]

    result = gather(specs, state, since)

    # both fetches had to be in flight together to clear the barrier, and output still
    # follows spec order regardless of which fetch finished first
    assert result == [a, b]


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


def test_run_source_task_drains_notices_from_state_into_result():
    # gather records source notices under a transient state key; the task must move them
    # onto the Result (for delivery) and clear them so they never persist to seen.json.
    state = {"ids": {}, "kv": {}, "_notices": ["x_src: creds expired"]}
    ctx = Context(
        state=state,
        gather=lambda specs, since: [],
        call=lambda *a, **k: "",
        logger=logging.getLogger("test"),
    )

    result = run_source_task(ctx, [_spec("k")], lambda ctx, items: "", "Subject")

    assert result.notices == ["x_src: creds expired"]
    assert "_notices" not in state
