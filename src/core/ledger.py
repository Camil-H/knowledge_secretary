# src/core/ledger.py
"""Per-model, per-UTC-day LLM request ledger persisted at state/llm_ledger.json.

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


# == Ledger ===================================================================


def load(path: str = PATH) -> Ledger:
    """Today's ledger; a missing, unreadable or stale-dated file yields a fresh day."""
    today = _today()
    if os.path.exists(path):
        try:
            with open(path) as f:
                data: Ledger = json.load(f)
        except (OSError, ValueError):
            logger.warning("⚠️ llm ledger at %s unreadable, starting a fresh day", path)
            data = {}
        if data.get("date") == today:
            data.setdefault("models", {})
            return data
    return {"date": today, "models": {}}


def consume(ledger: Ledger, model: str, *, path: str = PATH) -> None:
    """Count a dispatched request. Never refunded: a failed attempt may still have counted
    against the provider's quota."""
    _record(ledger, model)["requests"] += 1
    _save(ledger, path)


def mark_exhausted(ledger: Ledger, model: str, *, path: str = PATH) -> None:
    """Retire a model for the rest of the UTC day (its daily quota answered with a 429)."""
    _record(ledger, model)["exhausted"] = True
    _save(ledger, path)


def available(ledger: Ledger, model: str, rpd: int) -> bool:
    """Whether the model still has requests left today and hasn't been retired."""
    record = ledger.get("models", {}).get(model, {})
    return not record.get("exhausted", False) and record.get("requests", 0) < rpd


# == Helper Functions =========================================================


def _record(ledger: Ledger, model: str) -> dict[str, Any]:
    return ledger.setdefault("models", {}).setdefault(model, {"requests": 0, "exhausted": False})


def _save(ledger: Ledger, path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2, sort_keys=True)


def _today() -> str:
    return datetime.now(UTC).date().isoformat()
