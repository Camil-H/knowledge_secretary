# src/core/ledger.py
"""Quota ledger persisted at state/llm_ledger.json: one bucket per metered quantity, each
carrying the period it counts.

A bucket's `period` is the calendar string of its own granularity — a UTC day for a model's
request count, a UTC month for the Cloud TTS character count — so load() can drop what has
rolled over without knowing which quantity it is looking at.

The file lives under state/ so the publish action commits it and the 06:30 digest job sees
the 06:00 podcast job's counts. Every mutation is written through: a crashed run must not
forget quota it already spent.

The ledger day is UTC while Google resets quota at midnight US-Pacific. Both daily jobs fall
inside one Google day, and the 429 -> exhausted rule corrects any drift. Single-threaded by
construction: every LLM call runs on the main thread.
"""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

type Ledger = dict[str, Any]

PATH = "state/llm_ledger.json"
BUCKETS = "buckets"
TTS_KEY = "cloud-tts"

DAY = "day"
MONTH = "month"

_PERIOD_FORMATS: dict[str, str] = {DAY: "%Y-%m-%d", MONTH: "%Y-%m"}

_REQUEST_FIELDS: dict[str, Any] = {"requests": 0, "exhausted": False}
_TTS_FIELDS: dict[str, Any] = {"chars": 0}


# == Ledger ===================================================================


def load(path: str = PATH) -> Ledger:
    """The ledger with every rolled-over bucket dropped; a missing or unreadable file yields
    an empty one.

    A bucket survives exactly while its `period` still names the present at its own
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
    stored = data.get(BUCKETS)
    if not isinstance(stored, dict):
        stored = {}
    live = {
        name: bucket
        for name, bucket in stored.items()
        if isinstance(bucket, dict) and _is_current(bucket.get("period"))
    }
    return {BUCKETS: live}


def consume(ledger: Ledger, model: str, *, path: str = PATH) -> None:
    """Count a dispatched request. Never refunded: a failed attempt may still have counted
    against the provider's quota."""
    bucket = _bucket(ledger, model, DAY, _REQUEST_FIELDS)
    bucket["requests"] = int(bucket.get("requests", 0)) + 1
    _save(ledger, path)


def mark_exhausted(ledger: Ledger, model: str, *, path: str = PATH) -> None:
    """Retire a model for the rest of the UTC day (its daily quota answered with a 429)."""
    _bucket(ledger, model, DAY, _REQUEST_FIELDS)["exhausted"] = True
    _save(ledger, path)


def consume_tts_chars(ledger: Ledger, chars: int, *, path: str = PATH) -> int:
    """Add chars to the current month's Cloud TTS bucket (write-through); returns the month's
    running total. A bucket left from an earlier month is reset, never carried forward."""
    bucket = _bucket(ledger, TTS_KEY, MONTH, _TTS_FIELDS)
    total = int(bucket.get("chars", 0)) + chars
    bucket["chars"] = total
    _save(ledger, path)
    return total


def available(ledger: Ledger, model: str, rpd: int) -> bool:
    """Whether the model still has requests left today and hasn't been retired."""
    bucket = ledger.get(BUCKETS, {}).get(model, {})
    return not bucket.get("exhausted", False) and bucket.get("requests", 0) < rpd


# == Helper Functions =========================================================


def _bucket(ledger: Ledger, name: str, granularity: str, fields: dict[str, Any]) -> dict[str, Any]:
    """The named bucket at its current period, freshly zeroed when the stored one has rolled
    over — a Ledger held in memory can outlive the period load() validated it against."""
    period = _current_period(granularity)
    buckets = ledger.setdefault(BUCKETS, {})
    bucket = buckets.get(name)
    if not isinstance(bucket, dict) or bucket.get("period") != period:
        bucket = {"period": period, **fields}
        buckets[name] = bucket
    return bucket


def _current_period(granularity: str) -> str:
    return datetime.now(UTC).strftime(_PERIOD_FORMATS[granularity])


def _is_current(period: object) -> bool:
    """Whether a stored period still names the present. The granularities produce distinct
    strings, so matching any of them identifies the bucket's own granularity too."""
    return isinstance(period, str) and any(
        period == _current_period(granularity) for granularity in _PERIOD_FORMATS
    )


def _save(ledger: Ledger, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)
