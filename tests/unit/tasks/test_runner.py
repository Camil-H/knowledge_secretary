# tests/unit/test_runner.py
"""gather() reaches sources/enrichers only through the module-level registry names,
so each is swapped for a local fake registry here -- no real fetcher/enricher runs.
run_source_task() is driven through a faked ctx.gather, so gather()'s own logic is
out of scope for those tests."""

import logging
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from src import config
from src.core.errors import AuthError
from src.core.models import Context, Item
from src.tasks import runner
from src.tasks.runner import gather, run_source_task


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
    return lambda spec, since: items


def _raiser(exc: Exception):
    """Stands in for either a fetcher or an enricher, hence the loose signature."""

    def _raise(*_args):
        raise exc

    return _raise


# ----- gather: dedup, lookback window, and the per-source breakdown -----


@pytest.mark.parametrize(
    "fetched_ids, seen_ids, stale_ids, expected_ids, expected_log",
    [
        ([], [], [], [], "0 new of 0 fetched (0 seen, 0 outside window)"),
        (["a", "b"], [], [], ["a", "b"], "2 new of 2 fetched (0 seen, 0 outside window)"),
        (["a", "b", "c"], ["a"], ["b"], ["c"], "1 new of 3 fetched (1 seen, 1 outside window)"),
        # an item both seen and stale is counted once, so the parts always sum to fetched
        (["a"], ["a"], ["a"], [], "0 new of 1 fetched (1 seen, 0 outside window)"),
    ],
    ids=["empty-source", "all-new", "mixed", "seen-and-stale"],
)
def test_gather_returns_new_items_in_window_and_logs_the_breakdown(
    monkeypatch, caplog, fetched_ids, seen_ids, stale_ids, expected_ids, expected_log
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
        result = gather([_spec("src_key")], state, since)

    assert [it.id for it in result] == expected_ids
    assert f"gather: src_key → {expected_log}" in caplog.text


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


def _blocked_enricher(signal: threading.Event):
    """Enricher that finishes only once another item's enricher has signalled; run
    sequentially the wait times out instead."""

    def _enrich(item: Item) -> Item:
        if not signal.wait(timeout=5):
            raise TimeoutError("enrichment never overlapped")
        return replace(item, text="blocked")

    return _enrich


def _signalling_enricher(signal: threading.Event):
    def _enrich(item: Item) -> Item:
        signal.set()
        return replace(item, text="signalled")

    return _enrich


def test_gather_enriches_concurrently_and_returns_items_in_spec_order(monkeypatch):
    since = datetime.now(UTC) - timedelta(hours=1)
    first, second = _item("blocked:1"), _item("signal:1")
    monkeypatch.setattr(
        runner,
        "sources",
        _FakeRegistry({"a": _fetcher([first]), "b": _fetcher([second])}),
    )
    signal = threading.Event()
    monkeypatch.setattr(
        runner,
        "enrichers",
        _FakeRegistry(
            {"blocked": _blocked_enricher(signal), "signalling": _signalling_enricher(signal)}
        ),
    )
    state = {"ids": {}, "kv": {}}
    specs = [
        _spec("a_src", kind="a", enrich=["blocked"]),
        _spec("b_src", kind="b", enrich=["signalling"]),
    ]

    result = gather(specs, state, since)

    # the first item's enricher could only finish after the second's, yet output order
    # is still spec order rather than completion order
    assert [it.id for it in result] == ["blocked:1", "signal:1"]
    assert [it.text for it in result] == ["blocked", "signalled"]


def test_gather_enricher_crash_keeps_the_item_unenriched_and_logs(monkeypatch, caplog):
    since = datetime.now(UTC) - timedelta(hours=1)
    monkeypatch.setattr(runner, "sources", _FakeRegistry({"rss": _fetcher([_item("rss:1")])}))
    monkeypatch.setattr(
        runner,
        "enrichers",
        _FakeRegistry(
            {
                "boom": _raiser(RuntimeError("enricher boom")),
                "upper": lambda it: replace(it, text=it.text.upper()),
            }
        ),
    )
    state = {"ids": {}, "kv": {}}

    with caplog.at_level(logging.ERROR, logger="src.tasks.runner"):
        result = gather([_spec("k", enrich=["boom", "upper"])], state, since)

    assert [it.text for it in result] == ["BODY"]  # later enrichers still run
    assert "enricher boom" in caplog.text


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

    def _fetch(spec, since):
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
    before = datetime.now(UTC) - timedelta(hours=config.LOOKBACK_HOURS)
    run_source_task(ctx, [_spec("k")], lambda ctx, items: "unused", "Subject")
    after = datetime.now(UTC) - timedelta(hours=config.LOOKBACK_HOURS)

    assert before <= captured["since"] <= after


# ----- run_source_task: produce + consume -----


@pytest.mark.parametrize(
    "gathered_ids, produce_return, expected_markdown, expected_produce_calls",
    [
        ([], "should not be reached", "", 0),
        (["a:1", "a:2"], "", "", 1),
        (["a:1", "a:2"], "# Digest", "# Digest", 1),
    ],
    ids=["no-items-skips-produce", "items-produce-empty", "items-produce-markdown"],
)
def test_run_source_task_produces_from_items_and_consumes_every_gathered_id(
    gathered_ids, produce_return, expected_markdown, expected_produce_calls
):
    items = [_item(item_id) for item_id in gathered_ids]
    calls = {"n": 0}

    def _produce(ctx, produced_items):
        calls["n"] += 1
        return produce_return

    ctx = Context(
        state={"ids": {}, "kv": {}},
        gather=lambda specs, since: items,
        call=lambda *a, **k: "",
        logger=logging.getLogger("test"),
    )

    result = run_source_task(ctx, [_spec("k")], _produce, "Subject")

    assert calls["n"] == expected_produce_calls
    assert result.markdown == expected_markdown
    assert result.consumed == gathered_ids


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
