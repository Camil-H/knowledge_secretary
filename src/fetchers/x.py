"""X/Twitter fetching via the `twitter-cli` tool; auth from TWITTER_AUTH_TOKEN + TWITTER_CT0."""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

_CLI = "twitter"
_LIST_KEYS = ("tweets", "data", "results")
_DEFAULT_LIMIT = 20


def recent_tweets(handle: str, *, limit: int = _DEFAULT_LIMIT) -> list[dict]:
    """Recent tweets via `twitter user-posts <handle> --max N --json`; [] on failure.

    #TODO: field names follow twitter-cli's SCHEMA.md — verify against a live run.
    """
    try:
        proc = subprocess.run(
            [_CLI, "user-posts", handle, "--max", str(limit), "--json"],
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
        logger.warning("⚠️ x: unexpected JSON shape, top-level keys=%s", list(data)[:10])
        return []
    return data if isinstance(data, list) else []
