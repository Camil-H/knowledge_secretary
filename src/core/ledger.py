# src/core/ledger.py
"""Quota ledger persisted at state/llm_ledger.json: one top-level entry per metered
consumer, each carrying the period it counts.

An entry's `period` is the calendar string of its own granularity — a UTC day for a model's
request count, a UTC month for the Cloud TTS character count — so load() can drop what has
rolled over without knowing which quantity it is looking at.

The file lives under state/ so the publish action commits it and a later job sees an earlier
job's counts. Every mutation is written through: a crashed run must not forget quota it
already spent.

The ledger day is UTC while Google resets quota at midnight US-Pacific. Both daily jobs fall
inside one Google day, and the 429 -> exhausted rule corrects any drift. Single-threaded by
construction: every LLM call runs on the main thread.
"""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from src import config

logger = logging.getLogger(__name__)

type Ledger = dict[str, Any]

TTS_KEY = "cloud-tts"

_DAY_FORMAT = "%Y-%m-%d"
_MONTH_FORMAT = "%Y-%m"

_REQUEST_FIELDS: dict[str, Any] = {"requests": 0, "exhausted": False}
_TTS_FIELDS: dict[str, Any] = {"chars": 0}


# == Ledger ===================================================================


def load(path: str = config.LEDGER_PATH) -> Ledger:
    """The ledger with every rolled-over entry dropped; a missing or unreadable file yields
    an empty one.

    An entry survives exactly while its `period` still names the present at its own
    granularity, so the monthly TTS budget outlives a midnight rollover that clears the day's
    request counts, with no special case for either."""
    data: Ledger = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            logger.warning("⚠️ llm ledger at %s unreadable, starting fresh", path)
    if not isinstance(data, dict):
        data = {}
    return {
        name: entry
        for name, entry in data.items()
        if isinstance(entry, dict) and _is_current(entry.get("period"))
    }


def consume(ledger: Ledger, model: str, *, path: str = config.LEDGER_PATH) -> None:
    """Count a dispatched request. Never refunded: a failed attempt may still have counted
    against the provider's quota."""
    entry = _entry(ledger, model, _today(), _REQUEST_FIELDS)
    entry["requests"] = int(entry.get("requests", 0)) + 1
    _save(ledger, path)


def mark_exhausted(ledger: Ledger, model: str, *, path: str = config.LEDGER_PATH) -> None:
    """Retire a model for the rest of the UTC day (its daily quota answered with a 429)."""
    _entry(ledger, model, _today(), _REQUEST_FIELDS)["exhausted"] = True
    _save(ledger, path)


def consume_tts_chars(ledger: Ledger, chars: int, *, path: str = config.LEDGER_PATH) -> int:
    """Add chars to the current month's Cloud TTS entry (write-through); returns the month's
    running total. An entry left from an earlier month is reset, never carried forward."""
    entry = _entry(ledger, TTS_KEY, _this_month(), _TTS_FIELDS)
    total = int(entry.get("chars", 0)) + chars
    entry["chars"] = total
    _save(ledger, path)
    return total


def available(ledger: Ledger, model: str, rpd: int) -> bool:
    """Whether the model still has requests left today and hasn't been retired."""
    entry = ledger.get(model, {})
    return not entry.get("exhausted", False) and entry.get("requests", 0) < rpd


# == Helper Functions =========================================================


def _entry(ledger: Ledger, name: str, period: str, fields: dict[str, Any]) -> dict[str, Any]:
    """The named entry at the given period, freshly zeroed when the stored one has rolled
    over — a Ledger held in memory can outlive the period load() validated it against."""
    entry = ledger.get(name)
    if not isinstance(entry, dict) or entry.get("period") != period:
        entry = {"period": period, **fields}
        ledger[name] = entry
    return entry


def _today() -> str:
    return datetime.now(UTC).strftime(_DAY_FORMAT)


def _this_month() -> str:
    return datetime.now(UTC).strftime(_MONTH_FORMAT)


def _is_current(period: object) -> bool:
    """Whether a stored period still names the present. The day and month strings never
    collide, so matching either one identifies the entry's own granularity too."""
    return period in (_today(), _this_month())


def _save(ledger: Ledger, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)
