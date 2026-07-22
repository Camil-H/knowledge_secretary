"""Dedup memory + tiny KV persisted as state/seen.json: {"ids": {id: date}, "kv": {}}."""

import json
import os
from datetime import UTC, datetime, timedelta

from .models import Item

DEFAULT_PATH = "state/seen.json"


def load(path: str = DEFAULT_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        data.setdefault("ids", {})
        data.setdefault("kv", {})
        return data
    return {"ids": {}, "kv": {}}


def is_new(state: dict, item: Item) -> bool:
    return item.id not in state["ids"]


def mark_ids(state: dict, ids: list[str]) -> None:
    today = datetime.now(UTC).date().isoformat()
    for item_id in ids:
        state["ids"][item_id] = today


def get_kv(state: dict, key: str, default=None):
    return state["kv"].get(key, default)


def set_kv(state: dict, key: str, value) -> None:
    state["kv"][key] = value


def prune(state: dict, days: int = 60) -> None:
    # ISO dates sort lexicographically == chronologically.
    cutoff = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()
    state["ids"] = {k: v for k, v in state["ids"].items() if v >= cutoff}


def save(state: dict, path: str = DEFAULT_PATH) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
