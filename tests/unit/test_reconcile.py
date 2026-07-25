"""Semantic union of state/history JSON on rebase conflict. Pure merge logic; the git
plumbing in main() is a thin subprocess wrapper driven by the composite action."""

import json

import pytest

from src.delivery import reconcile
from src.delivery.reconcile import main, merge_history_entry, merge_state

# ----- merge_history_entry -----


def test_merge_history_entry_unions_task_cards():
    a = {"date": "2026-07-25", "tasks": {"newsletter": {"x": 1}, "youtube": {"y": 2}}}
    b = {"date": "2026-07-25", "tasks": {"podcast": {"z": 3}}}

    merged = merge_history_entry(a, b)

    assert set(merged["tasks"]) == {"newsletter", "youtube", "podcast"}
    assert merged["date"] == "2026-07-25"


def test_merge_history_entry_tolerates_a_missing_side():
    b = {"date": "2026-07-25", "tasks": {"podcast": {"z": 3}}}
    assert merge_history_entry(None, b)["tasks"] == {"podcast": {"z": 3}}


# ----- merge_state: ids -----


def test_merge_state_unions_ids():
    base = {"ids": {"a:1": "d"}, "kv": {}}
    a = {"ids": {"a:1": "d", "b:2": "d"}, "kv": {}}
    b = {"ids": {"a:1": "d", "c:3": "d"}, "kv": {}}

    assert set(merge_state(base, a, b)["ids"]) == {"a:1", "b:2", "c:3"}


# ----- merge_state: kv 3-way -----


@pytest.mark.parametrize(
    "a_queue,b_queue,expected",
    [
        (["Y"], ["X", "Y"], ["Y"]),  # a advanced the queue, b untouched -> a
        (["X", "Y"], ["Y"], ["Y"]),  # b advanced, a untouched -> b
        (["X", "Y"], ["X", "Y"], ["X", "Y"]),  # neither changed -> unchanged
    ],
    ids=["a-changed", "b-changed", "neither"],
)
def test_merge_state_kv_keeps_the_side_that_changed(a_queue, b_queue, expected):
    base = {"ids": {}, "kv": {"podcast_queue": ["X", "Y"]}}
    a = {"ids": {}, "kv": {"podcast_queue": a_queue}}
    b = {"ids": {}, "kv": {"podcast_queue": b_queue}}

    assert merge_state(base, a, b)["kv"]["podcast_queue"] == expected


# ----- main: path routing -----


def test_main_rejects_unexpected_conflicted_path(monkeypatch):
    monkeypatch.setattr(reconcile, "_conflicted_paths", lambda: ["src/core/llm.py"])
    assert main() == 1


def test_main_unions_conflicted_history_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "history").mkdir()
    path = "history/2026-07-25.json"
    monkeypatch.setattr(reconcile, "_conflicted_paths", lambda: [path])
    stages = {
        2: {"date": "2026-07-25", "tasks": {"newsletter": {"x": 1}}},
        3: {"date": "2026-07-25", "tasks": {"podcast": {"z": 3}}},
    }
    monkeypatch.setattr(reconcile, "_blob", lambda stage, _p: stages.get(stage))
    added = []
    monkeypatch.setattr(reconcile.subprocess, "run", lambda argv, **k: added.append(argv))

    assert main() == 0
    written = json.loads((tmp_path / path).read_text())
    assert set(written["tasks"]) == {"newsletter", "podcast"}
    assert added == [["git", "add", path]]
