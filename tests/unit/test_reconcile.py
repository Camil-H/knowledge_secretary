"""Semantic union of state/history JSON on rebase conflict. Pure merge logic; the git
plumbing in main() is a thin subprocess wrapper driven by the composite action."""

import json

import pytest

from src.delivery import reconcile
from src.delivery.reconcile import (
    LEDGER_PATH,
    main,
    merge_history_entry,
    merge_ledger,
    merge_state,
)

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


# ----- merge_ledger -----

_MODEL = "gemini-3.6-flash"
_DAY = "2026-07-28"


def _ledger(requests: int, *, exhausted: bool = False, date: str = _DAY) -> dict:
    return {"date": date, "models": {_MODEL: {"requests": requests, "exhausted": exhausted}}}


@pytest.mark.parametrize(
    "base, a, b, expected_requests",
    [
        pytest.param(_ledger(2), _ledger(5), _ledger(9), 12, id="additive_from_base"),
        pytest.param(None, _ledger(5), _ledger(9), 14, id="missing_base_sums_both_sides"),
        pytest.param(_ledger(5), _ledger(5), _ledger(5), 5, id="neither_side_moved"),
    ],
)
def test_merge_ledger_adds_each_sides_new_requests(base, a, b, expected_requests):
    """Undercounting would cost at most one extra 429, but overcounting would waste budget."""
    assert merge_ledger(base, a, b)["models"][_MODEL]["requests"] == expected_requests


@pytest.mark.parametrize(
    "a_exhausted, b_exhausted, expected",
    [(False, False, False), (True, False, True), (False, True, True), (True, True, True)],
)
def test_merge_ledger_ors_exhaustion(a_exhausted, b_exhausted, expected):
    a, b = _ledger(1, exhausted=a_exhausted), _ledger(1, exhausted=b_exhausted)
    assert merge_ledger(_ledger(0), a, b)["models"][_MODEL]["exhausted"] is expected


def test_merge_ledger_keeps_the_newer_day_when_the_dates_differ():
    older = {"date": "2026-07-27", "models": {_MODEL: {"requests": 9}}}
    newer = {"date": _DAY, "models": {_MODEL: {"requests": 2}}}
    assert merge_ledger(None, older, newer) == newer


@pytest.mark.parametrize(
    "base_chars, a_chars, b_chars, expected",
    [
        pytest.param(1000, 1500, 1800, 2300, id="additive_from_base"),
        pytest.param(None, 1500, 1800, 3300, id="missing_base_sums_both_sides"),
        pytest.param(1500, 1500, 1500, 1500, id="neither_side_moved"),
    ],
)
def test_merge_ledger_adds_each_sides_new_tts_chars(base_chars, a_chars, b_chars, expected):
    month = _DAY[:7]

    def _with_tts(chars):
        return {**_ledger(1), "tts": {"month": month, "chars": chars}}

    base = None if base_chars is None else _with_tts(base_chars)
    merged = merge_ledger(base, _with_tts(a_chars), _with_tts(b_chars))
    assert merged["tts"] == {"month": month, "chars": expected}


def test_merge_ledger_merges_tts_on_its_own_month_across_a_day_rollover():
    """Both jobs of a month can straddle midnight; the characters still belong to one month."""
    month = _DAY[:7]
    a = {"date": "2026-07-27", "models": {}, "tts": {"month": month, "chars": 500}}
    b = {"date": _DAY, "models": {}, "tts": {"month": month, "chars": 700}}

    merged = merge_ledger(None, a, b)
    assert merged["date"] == _DAY
    assert merged["tts"]["chars"] == 1200


def test_merge_ledger_keeps_the_newer_month_when_the_tts_months_differ():
    a = {"date": _DAY, "models": {}, "tts": {"month": "2026-06", "chars": 900_000}}
    b = {"date": _DAY, "models": {}, "tts": {"month": "2026-07", "chars": 1_000}}
    assert merge_ledger(None, a, b)["tts"] == {"month": "2026-07", "chars": 1_000}


def test_merge_ledger_keeps_a_tts_bucket_only_one_side_has():
    bucket = {"month": "2026-07", "chars": 33}
    a = {"date": _DAY, "models": {}, "tts": bucket}
    b = {"date": _DAY, "models": {}}
    assert merge_ledger(None, a, b)["tts"] == bucket


def test_merge_ledger_omits_tts_when_neither_side_has_one():
    assert "tts" not in merge_ledger(None, _ledger(1), _ledger(2))


def test_merge_ledger_keeps_models_only_one_side_has():
    a = {"date": _DAY, "models": {_MODEL: {"requests": 3}}}
    b = {"date": _DAY, "models": {"gemini-3.1-flash": {"requests": 7}}}
    merged = merge_ledger(None, a, b)
    assert merged["models"][_MODEL]["requests"] == 3
    assert merged["models"]["gemini-3.1-flash"]["requests"] == 7


def test_merge_ledger_keeps_the_newer_day_whole_across_a_rollover():
    """A UTC-day rollover mid-conflict means the older record is spent quota from yesterday."""
    yesterday, today = _ledger(19, date="2026-07-27"), _ledger(2)
    assert merge_ledger(yesterday, yesterday, today) == today
    assert merge_ledger(yesterday, today, yesterday) == today


@pytest.mark.parametrize("side", ["a", "b"], ids=["theirs_missing", "ours_missing"])
def test_merge_ledger_tolerates_a_missing_side(side):
    present = _ledger(4)
    args = (present, None) if side == "a" else (None, present)
    assert merge_ledger(None, *args) == present


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


def test_main_merges_a_conflicted_ledger(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "state").mkdir()
    monkeypatch.setattr(reconcile, "_conflicted_paths", lambda: [LEDGER_PATH])
    stages = {1: _ledger(1), 2: _ledger(4), 3: _ledger(6)}
    monkeypatch.setattr(reconcile, "_blob", lambda stage, _p: stages.get(stage))
    monkeypatch.setattr(reconcile.subprocess, "run", lambda argv, **k: None)

    assert main() == 0
    written = json.loads((tmp_path / LEDGER_PATH).read_text())
    assert written["models"][_MODEL]["requests"] == 9
