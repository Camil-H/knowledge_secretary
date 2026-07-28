"""Resolve rebase conflicts in the committed state/history JSON by semantic union.

The daily-digest jobs (podcast, and newsletter+youtube) both write the same day's
history file and the shared dedup state; a git text rebase can't merge those, so on
conflict this unions them — every run's output survives. Invoked from the publish
composite action's commit step when `git rebase` reports conflicts.
"""

import json
import subprocess
import sys

from src.core.ledger import PATH as LEDGER_PATH

STATE_PATH = "state/seen.json"
_EMPTY_STATE: dict = {"ids": {}, "kv": {}}


# == Merge ====================================================================


def merge_history_entry(a: dict | None, b: dict | None) -> dict:
    """Union two versions of a day's entry; task cards are partitioned by task, so a
    key-wise union never drops either run's card."""
    a, b = a or {}, b or {}
    return {
        "date": a.get("date") or b.get("date"),
        "tasks": {**(b.get("tasks") or {}), **(a.get("tasks") or {})},
    }


def merge_state(base: dict | None, a: dict | None, b: dict | None) -> dict:
    """3-way merge of seen.json: ids are append-only (union); each kv key is mutated by
    at most one run, so keep whichever side changed it from base."""
    base, a, b = base or _EMPTY_STATE, a or _EMPTY_STATE, b or _EMPTY_STATE
    ids = {**b.get("ids", {}), **a.get("ids", {})}
    return {"ids": ids, "kv": _merge_kv(base.get("kv", {}), a.get("kv", {}), b.get("kv", {}))}


def merge_ledger(base: dict | None, a: dict | None, b: dict | None) -> dict:
    """3-way merge of the LLM request ledger.

    Same day: request counts are additive (`a + b - base`, floored by each side so a missing
    base can't undercount) and `exhausted` is an OR — an undercount would cost at most one
    extra 429, which itself retires the model. Different days: one side already rolled over,
    so the newer record wins whole."""
    a, b = a or {}, b or {}
    if not a or not b:
        return a or b
    if a.get("date") != b.get("date"):
        return a if (a.get("date") or "") >= (b.get("date") or "") else b

    base_models = (base or {}).get("models", {})
    a_models, b_models = a.get("models", {}), b.get("models", {})
    models = {}
    for model in a_models.keys() | b_models.keys():
        a_requests = a_models.get(model, {}).get("requests", 0)
        b_requests = b_models.get(model, {}).get("requests", 0)
        base_requests = base_models.get(model, {}).get("requests", 0)
        models[model] = {
            "requests": max(a_requests, b_requests, a_requests + b_requests - base_requests),
            "exhausted": bool(
                a_models.get(model, {}).get("exhausted") or b_models.get(model, {}).get("exhausted")
            ),
        }
    return {"date": a.get("date"), "models": models}


def _merge_kv(base: dict, a: dict, b: dict) -> dict:
    merged = {}
    for key in base.keys() | a.keys() | b.keys():
        a_changed = key in a and a[key] != base.get(key)
        b_changed = key in b and b[key] != base.get(key)
        if b_changed and not a_changed:
            merged[key] = b[key]
        else:
            merged[key] = a[key] if key in a else b[key]
    return merged


# == Entry point ==============================================================


def _blob(stage: int, path: str) -> dict | None:
    proc = subprocess.run(["git", "show", f":{stage}:{path}"], capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return json.loads(proc.stdout)


def _conflicted_paths() -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def main() -> int:
    for path in _conflicted_paths():
        base, ours, theirs = _blob(1, path), _blob(2, path), _blob(3, path)
        if path.startswith("history/"):
            merged = merge_history_entry(ours, theirs)
        elif path == STATE_PATH:
            merged = merge_state(base, ours, theirs)
        elif path == LEDGER_PATH:
            merged = merge_ledger(base, ours, theirs)
        else:
            print(f"reconcile: refusing to auto-merge unexpected path {path}", file=sys.stderr)
            return 1
        with open(path, "w") as f:
            json.dump(merged, f, indent=2, sort_keys=True)
        subprocess.run(["git", "add", path], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
