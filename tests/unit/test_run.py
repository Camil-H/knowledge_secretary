"""_run_task() and main() reach the registries and state module only through the
module-level names imported in src.run, so each is swapped for a local fake here --
no real task/deliverer/state I/O runs. See tests/e2e/test_pipeline.py for the
happy-path, fully-wired coverage of the same entrypoint."""

import pytest

import src.run as run
from src.core.models import Result


class _FakeRegistry:
    """Minimal name->callable lookup mirroring src.core.registry.Registry.get
    (KeyError on a miss, same as the real registry)."""

    def __init__(self, mapping: dict | None = None):
        self._d = dict(mapping or {})

    def get(self, name: str):
        return self._d[name]


def _task(result: Result):
    return lambda ctx: result


def _deliverer(sink: dict):
    def _deliver(result):
        sink["result"] = result

    return _deliver


def _raiser(exc: Exception):
    def _raise(*args, **kwargs):
        raise exc

    return _raise


# ----- _run_task: consumed ids only burned after successful delivery -----


def test_run_task_deliverer_raises_leaves_mark_ids_uncalled(monkeypatch):
    """Only the produce-stage failure is covered e2e; this pins the same invariant
    for a delivery-stage failure -- a raising deliverer must not burn consumed ids."""
    result = Result(consumed=["a:1", "a:2"])
    monkeypatch.setattr(run, "tasks", _FakeRegistry({"mytask": _task(result)}))
    monkeypatch.setattr(run, "deliverers", _FakeRegistry({"site": _raiser(RuntimeError("boom"))}))
    mark_calls = []
    monkeypatch.setattr(run.state_mod, "mark_ids", lambda state, ids: mark_calls.append(ids))
    state = {"ids": {}, "kv": {}}

    with pytest.raises(RuntimeError, match="boom"):
        run._run_task("mytask", state)

    assert mark_calls == []


# ----- _run_task: meta['task'] setdefault -----


@pytest.mark.parametrize(
    "initial_meta, expected_task",
    [
        ({}, "mytask"),
        ({"task": "explicit-task"}, "explicit-task"),
    ],
    ids=["absent-defaults-to-task-name", "explicit-value-preserved"],
)
def test_run_task_meta_task_is_setdefault_not_overwrite(monkeypatch, initial_meta, expected_task):
    result = Result(meta=dict(initial_meta))
    delivered = {}
    monkeypatch.setattr(run, "tasks", _FakeRegistry({"mytask": _task(result)}))
    monkeypatch.setattr(run, "deliverers", _FakeRegistry({"site": _deliverer(delivered)}))
    monkeypatch.setattr(run.state_mod, "mark_ids", lambda state, ids: None)
    state = {"ids": {}, "kv": {}}

    run._run_task("mytask", state)

    assert delivered["result"].meta["task"] == expected_task


# ----- main: unknown task name -----


def test_main_unknown_task_name_is_caught_per_task_and_returns_1(monkeypatch):
    monkeypatch.setattr(run.state_mod, "load", lambda: {"ids": {}, "kv": {}})
    pruned = {}
    saved = {}
    monkeypatch.setattr(run.state_mod, "prune", lambda state: pruned.setdefault("called", True))
    monkeypatch.setattr(run.state_mod, "save", lambda state: saved.setdefault("called", True))
    monkeypatch.setattr(run, "tasks", _FakeRegistry())  # nothing registered -> KeyError on .get

    code = run.main(["prog", "no-such-task"])

    assert code == 1
    assert pruned["called"]  # loop's KeyError didn't skip the post-loop prune/save
    assert saved["called"]
