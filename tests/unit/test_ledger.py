"""Period-scoped quota buckets. Every case runs against a tmp_path file, so the committed
state/llm_ledger.json is never touched."""

import json
from datetime import UTC, datetime

import pytest

from src.core import ledger as ledger_mod

_MODEL = "gemini-3.6-flash"
_BUCKETS = ledger_mod.BUCKETS


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


def _day_bucket(requests: int, *, exhausted: bool = False, period: str | None = None) -> dict:
    return {"period": period or _today(), "requests": requests, "exhausted": exhausted}


# ----- load -----


def test_load_starts_empty_when_the_file_is_missing(tmp_path):
    assert ledger_mod.load(_path(tmp_path)) == {_BUCKETS: {}}


def test_load_keeps_todays_counts(tmp_path):
    path = _path(tmp_path)
    ledger_mod.consume(ledger_mod.load(path), _MODEL, path=path)
    assert ledger_mod.load(path)[_BUCKETS][_MODEL]["requests"] == 1


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param({_BUCKETS: {_MODEL: _day_bucket(19, period="2000-01-01")}}, id="stale_day"),
        pytest.param({_BUCKETS: {_MODEL: {"requests": 19}}}, id="no_period"),
        pytest.param({_BUCKETS: {_MODEL: "not a bucket"}}, id="not_a_bucket"),
        pytest.param({_BUCKETS: "not a mapping"}, id="buckets_not_a_mapping"),
        pytest.param({"not": "a ledger"}, id="foreign_shape"),
    ],
)
def test_load_drops_a_bucket_that_is_not_of_the_current_period(tmp_path, stored):
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    _write(path, stored)
    assert ledger_mod.load(path) == {_BUCKETS: {}}


def test_load_degrades_to_an_empty_ledger_on_an_unreadable_file(tmp_path):
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    with open(path, "w") as f:
        f.write("{not json")
    assert ledger_mod.load(path) == {_BUCKETS: {}}


def test_load_keeps_each_bucket_on_its_own_period(tmp_path):
    """The day's requests and the month's characters roll over independently: one file, one
    pruning rule, two granularities."""
    path = _path(tmp_path)
    (tmp_path / "state").mkdir()
    tts = {"period": _month(), "chars": 12_345}
    _write(
        path,
        {
            _BUCKETS: {
                _MODEL: _day_bucket(3, period="2000-01-01"),
                "fresh": _day_bucket(1),
                ledger_mod.TTS_KEY: tts,
                "old-tts": {"period": "1999-01", "chars": 9},
            }
        },
    )

    assert ledger_mod.load(path) == {_BUCKETS: {"fresh": _day_bucket(1), ledger_mod.TTS_KEY: tts}}


# ----- consume / mark_exhausted -----


def test_consume_increments_and_writes_through(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    dispatches = 3
    for _ in range(dispatches):
        ledger_mod.consume(ledger, _MODEL, path=path)

    assert ledger[_BUCKETS][_MODEL] == _day_bucket(dispatches)
    assert _read(path)[_BUCKETS][_MODEL]["requests"] == dispatches


def test_consume_counts_each_model_separately(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    other = "gemini-3.1-flash-lite"
    ledger_mod.consume(ledger, _MODEL, path=path)
    ledger_mod.consume(ledger, other, path=path)

    counts = {name: bucket["requests"] for name, bucket in _read(path)[_BUCKETS].items()}
    assert counts == {_MODEL: 1, other: 1}


def test_consume_resets_a_bucket_left_from_an_earlier_day(tmp_path):
    """An in-memory Ledger can outlive the day load() validated it against."""
    path = _path(tmp_path)
    ledger = {_BUCKETS: {_MODEL: _day_bucket(19, exhausted=True, period="2000-01-01")}}
    ledger_mod.consume(ledger, _MODEL, path=path)

    assert ledger[_BUCKETS][_MODEL] == _day_bucket(1)


def test_mark_exhausted_persists(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    ledger_mod.mark_exhausted(ledger, _MODEL, path=path)

    assert _read(path)[_BUCKETS][_MODEL]["exhausted"] is True
    assert not ledger_mod.available(ledger_mod.load(path), _MODEL, rpd=99)


# ----- consume_tts_chars -----


def test_consume_tts_chars_accumulates_and_writes_through(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    first = ledger_mod.consume_tts_chars(ledger, 400, path=path)
    total = ledger_mod.consume_tts_chars(ledger, 600, path=path)

    assert (first, total) == (400, 1000)
    assert _read(path)[_BUCKETS][ledger_mod.TTS_KEY] == {"period": _month(), "chars": 1000}


def test_consume_tts_chars_resets_a_bucket_from_an_earlier_month(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    ledger[_BUCKETS][ledger_mod.TTS_KEY] = {"period": "1999-01", "chars": 900_000}

    assert ledger_mod.consume_tts_chars(ledger, 500, path=path) == 500
    assert ledger[_BUCKETS][ledger_mod.TTS_KEY]["period"] == _month()


def test_consume_tts_chars_leaves_the_model_counts_alone(tmp_path):
    path = _path(tmp_path)
    ledger = ledger_mod.load(path)
    ledger_mod.consume(ledger, _MODEL, path=path)
    ledger_mod.consume_tts_chars(ledger, 42, path=path)

    stored = _read(path)[_BUCKETS]
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
    buckets = {} if requests is None else {_MODEL: _day_bucket(requests, exhausted=exhausted)}
    assert ledger_mod.available({_BUCKETS: buckets}, _MODEL, rpd) is expected


@pytest.mark.parametrize("spent", [0, 1])
def test_available_is_false_once_the_daily_budget_is_spent(spent):
    rpd = 3
    ledger = {_BUCKETS: {_MODEL: _day_bucket(rpd + spent)}}
    assert ledger_mod.available(ledger, _MODEL, rpd) is False
