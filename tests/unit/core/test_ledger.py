"""Period-scoped quota entries. Every case runs against a tmp_path file, so the committed
state/llm_ledger.json is never touched."""

import json
from datetime import UTC, datetime

import pytest

from src.core import ledger as ledger_mod

_MODEL = "gemini-3.6-flash"


def _path(tmp_path):
    return str(tmp_path / "state" / "llm_ledger.json")


def _write(path: str, payload: object) -> None:
    with open(path, "w") as f:
        json.dump(payload, f)


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _day_entry(requests: int, *, exhausted: bool = False, period: str | None = None) -> dict:
    return {"period": period or _today(), "requests": requests, "exhausted": exhausted}


# ----- load -----


def test_load_starts_empty_when_the_file_is_missing(tmp_path):
    assert ledger_mod.load(_path(tmp_path)) == {}


def test_load_keeps_todays_counts(tmp_path):
    path = _path(tmp_path)
    ledger_mod.consume(ledger_mod.load(path), _MODEL, path=path)
    assert ledger_mod.load(path)[_MODEL]["requests"] == 1


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param({_MODEL: _day_entry(19, period="2000-01-01")}, id="stale_day"),
        pytest.param({_MODEL: {"requests": 19}}, id="no_period"),
        pytest.param({_MODEL: "not an entry"}, id="not_an_entry"),
        pytest.param({"not": "a ledger"}, id="foreign_shape"),
        pytest.param([{"period": _today(), "requests": 1}], id="file_not_a_mapping"),
    ],
)
def test_load_drops_an_entry_that_is_not_of_the_current_period(tmp_path, stored):
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    _write(path, stored)
    assert ledger_mod.load(path) == {}


def test_load_degrades_to_an_empty_ledger_on_an_unreadable_file(tmp_path):
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    with open(path, "w") as f:
        f.write("{not json")
    assert ledger_mod.load(path) == {}


def test_load_keeps_each_entry_on_its_own_period(tmp_path):
    """The day's requests and the month's characters roll over independently: one file, one
    pruning rule, two granularities."""
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    tts = {"period": _month(), "chars": 12_345}
    _write(
        path,
        {
            _MODEL: _day_entry(3, period="2000-01-01"),
            "fresh": _day_entry(1),
            ledger_mod.TTS_KEY: tts,
            "old-tts": {"period": "1999-01", "chars": 9},
        },
    )

    assert ledger_mod.load(path) == {"fresh": _day_entry(1), ledger_mod.TTS_KEY: tts}


# ----- consume / mark_exhausted -----


def test_consume_increments_and_writes_through(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    dispatches = 3
    for _ in range(dispatches):
        ledger_mod.consume(ledger, _MODEL, path=path)

    assert ledger[_MODEL] == _day_entry(dispatches)
    assert _read(path)[_MODEL]["requests"] == dispatches


def test_consume_counts_each_model_separately(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    other = "gemini-3.1-flash-lite"
    ledger_mod.consume(ledger, _MODEL, path=path)
    ledger_mod.consume(ledger, other, path=path)

    counts = {name: entry["requests"] for name, entry in _read(path).items()}
    assert counts == {_MODEL: 1, other: 1}


def test_consume_resets_an_entry_left_from_an_earlier_day(tmp_path):
    """An in-memory Ledger can outlive the day load() validated it against."""
    path = _path(tmp_path)
    ledger = {_MODEL: _day_entry(19, exhausted=True, period="2000-01-01")}
    ledger_mod.consume(ledger, _MODEL, path=path)

    assert ledger[_MODEL] == _day_entry(1)


def test_mark_exhausted_persists(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    ledger_mod.mark_exhausted(ledger, _MODEL, path=path)

    assert _read(path)[_MODEL]["exhausted"] is True
    assert not ledger_mod.available(ledger_mod.load(path), _MODEL, rpd=99)


# ----- consume_tts_chars -----


def test_consume_tts_chars_accumulates_and_writes_through(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    first = ledger_mod.consume_tts_chars(ledger, 400, path=path)
    total = ledger_mod.consume_tts_chars(ledger, 600, path=path)

    assert (first, total) == (400, 1000)
    assert _read(path)[ledger_mod.TTS_KEY] == {"period": _month(), "chars": 1000}


def test_consume_tts_chars_resets_an_entry_from_an_earlier_month(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    ledger[ledger_mod.TTS_KEY] = {"period": "1999-01", "chars": 900_000}

    assert ledger_mod.consume_tts_chars(ledger, 500, path=path) == 500
    assert ledger[ledger_mod.TTS_KEY]["period"] == _month()


def test_consume_tts_chars_leaves_the_model_counts_alone(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    ledger_mod.consume(ledger, _MODEL, path=path)
    ledger_mod.consume_tts_chars(ledger, 42, path=path)

    stored = _read(path)
    assert stored[_MODEL]["requests"] == 1
    assert stored[ledger_mod.TTS_KEY]["chars"] == 42


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
    ledger = {} if requests is None else {_MODEL: _day_entry(requests, exhausted=exhausted)}
    assert ledger_mod.available(ledger, _MODEL, rpd) is expected


@pytest.mark.parametrize("spent", [0, 1])
def test_available_is_false_once_the_daily_budget_is_spent(spent):
    rpd = 3
    ledger = {_MODEL: _day_entry(rpd + spent)}
    assert ledger_mod.available(ledger, _MODEL, rpd) is False
