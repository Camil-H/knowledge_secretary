"""Per-model per-UTC-day request ledger. Every case runs against a tmp_path file, so the
committed state/llm_ledger.json is never touched."""

import json
from datetime import UTC, datetime

import pytest

from src.core import ledger as ledger_mod

_MODEL = "gemini-3.6-flash"


def _path(tmp_path):
    return str(tmp_path / "state" / "llm_ledger.json")


def _write(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f)


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ----- load -----


def test_load_starts_a_fresh_day_when_the_file_is_missing(tmp_path):
    assert ledger_mod.load(_path(tmp_path)) == {"date": _today(), "models": {}}


def test_load_keeps_todays_counts(tmp_path):
    path = _path(tmp_path)
    ledger_mod.consume(ledger_mod.load(path), _MODEL, path=path)
    assert ledger_mod.load(path)["models"][_MODEL]["requests"] == 1


@pytest.mark.parametrize(
    "stored",
    [
        {"date": "2000-01-01", "models": {_MODEL: {"requests": 19, "exhausted": True}}},
        {"not": "a ledger"},
    ],
    ids=["stale_date", "no_date"],
)
def test_load_rolls_over_when_the_stored_date_is_not_today(tmp_path, stored):
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    _write(path, stored)
    assert ledger_mod.load(path) == {"date": _today(), "models": {}}


def test_load_degrades_to_a_fresh_day_on_an_unreadable_file(tmp_path):
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    with open(path, "w") as f:
        f.write("{not json")
    assert ledger_mod.load(path) == {"date": _today(), "models": {}}


# ----- consume / mark_exhausted -----


def test_consume_increments_and_writes_through(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    dispatches = 3
    for _ in range(dispatches):
        ledger_mod.consume(ledger, _MODEL, path=path)

    assert ledger["models"][_MODEL]["requests"] == dispatches
    assert _read(path)["models"][_MODEL]["requests"] == dispatches


def test_consume_counts_each_model_separately(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    other = "gemini-3.1-flash"
    ledger_mod.consume(ledger, _MODEL, path=path)
    ledger_mod.consume(ledger, other, path=path)

    counts = {model: record["requests"] for model, record in _read(path)["models"].items()}
    assert counts == {_MODEL: 1, other: 1}


def test_mark_exhausted_persists(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    ledger_mod.mark_exhausted(ledger, _MODEL, path=path)

    assert _read(path)["models"][_MODEL]["exhausted"] is True
    assert not ledger_mod.available(ledger_mod.load(path), _MODEL, rpd=99)


# ----- consume_tts_chars -----


def test_consume_tts_chars_accumulates_and_writes_through(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    first = ledger_mod.consume_tts_chars(ledger, 400, path=path)
    total = ledger_mod.consume_tts_chars(ledger, 600, path=path)

    assert (first, total) == (400, 1000)
    assert _read(path)[ledger_mod.TTS_KEY] == {"month": _month(), "chars": 1000}


def test_consume_tts_chars_resets_a_bucket_from_an_earlier_month(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    ledger[ledger_mod.TTS_KEY] = {"month": "1999-01", "chars": 900_000}

    assert ledger_mod.consume_tts_chars(ledger, 500, path=path) == 500
    assert ledger[ledger_mod.TTS_KEY]["month"] == _month()


def test_consume_tts_chars_leaves_the_model_counts_alone(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    ledger_mod.consume(ledger, _MODEL, path=path)
    ledger_mod.consume_tts_chars(ledger, 42, path=path)

    stored = _read(path)
    assert stored["models"][_MODEL]["requests"] == 1
    assert stored[ledger_mod.TTS_KEY]["chars"] == 42


def test_load_keeps_this_months_tts_bucket_across_a_day_rollover(tmp_path):
    """The character budget is monthly: dropping it at midnight would lose most of a month."""
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    bucket = {"month": _month(), "chars": 12_345}
    _write(path, {"date": "2000-01-01", "models": {_MODEL: {"requests": 3}}, "tts": bucket})

    loaded = ledger_mod.load(path)
    assert loaded == {"date": _today(), "models": {}, "tts": bucket}


def test_load_drops_a_tts_bucket_from_an_earlier_month(tmp_path):
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    _write(path, {"date": "2000-01-01", "models": {}, "tts": {"month": "1999-01", "chars": 9}})

    assert ledger_mod.load(path) == {"date": _today(), "models": {}}


# ----- available -----


@pytest.mark.parametrize(
    "requests, exhausted, expected",
    [
        (0, False, True),
        (1, False, True),
        (0, True, False),
        (None, False, True),
    ],
    ids=["unused", "part_spent", "retired", "unknown_model"],
)
def test_available_matrix(requests, exhausted, expected):
    rpd = 4
    models = {} if requests is None else {_MODEL: {"requests": requests, "exhausted": exhausted}}
    ledger = {"date": _today(), "models": models}
    assert ledger_mod.available(ledger, _MODEL, rpd) is expected


@pytest.mark.parametrize("spent", [0, 1])
def test_available_is_false_once_the_daily_budget_is_spent(spent):
    rpd = 3
    ledger = {
        "date": _today(),
        "models": {_MODEL: {"requests": rpd + spent, "exhausted": False}},
    }
    assert ledger_mod.available(ledger, _MODEL, rpd) is False
