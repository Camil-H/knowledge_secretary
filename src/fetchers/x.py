"""X/Twitter fetching via the agent-reach `twitter` (twitter-cli) backend."""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

_CLI = "twitter"
_LIST_KEYS = ("tweets", "data", "results")
_DEFAULT_LIMIT = 20


def recent_tweets(handle: str, *, limit: int = _DEFAULT_LIMIT) -> list[dict]:
    """Recent tweets for a handle via `twitter search "from:<handle>" -n N --json`.

    Best-effort: the backend is frequently unavailable (stale X cookie / not
    installed), so any failure degrades to []. Each tweet dict carries whatever
    the CLI emits (commonly id/text/url/created_at).

    #TODO: subcommand/flags are from agent-reach's docs but the JSON output shape
    is UNVERIFIED — confirm with one live run and adjust `_extract` if needed.
    """
    try:
        proc = subprocess.run(
            [_CLI, "search", f"from:{handle}", "-n", str(limit), "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return _extract(json.loads(proc.stdout))
    except Exception as e:
        logger.warning("⚠️ x %s degraded (no items): %s", handle, e)
        return []


# == Helper Functions =========================================================


def _extract(data) -> list[dict]:
    """Tolerate a top-level JSON array or an object wrapping the tweet list."""
    if isinstance(data, dict):
        for key in _LIST_KEYS:
            if isinstance(data.get(key), list):
                return data[key]
        return []
    return data if isinstance(data, list) else []
