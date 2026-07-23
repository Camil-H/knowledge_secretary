"""X/Twitter fetching via the `twitter-cli` tool; auth from TWITTER_AUTH_TOKEN + TWITTER_CT0."""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

_CLI = "twitter"
_LIST_KEYS = ("tweets", "data", "results")
_DEFAULT_LIMIT = 20


# == Exceptions ===============================================================


class UnexpectedXFormat(Exception):
    """twitter-cli output didn't match the expected tweet schema."""


# == Fetch ====================================================================


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
    except UnexpectedXFormat:
        raise  # format drift is loud, not a silent degrade
    except Exception as e:
        logger.warning("⚠️ x %s degraded (no items): %s", handle, e)
        return []


# == Helper Functions =========================================================


def _extract(data) -> list[dict]:
    """Return the tweet list from a top-level array or a known wrapper key."""
    if isinstance(data, dict):
        for key in _LIST_KEYS:
            if isinstance(data.get(key), list):
                return data[key]
        raise UnexpectedXFormat(f"no tweet list in response; keys={list(data)[:10]}")
    if isinstance(data, list):
        return data
    raise UnexpectedXFormat(f"expected a list or object, got {type(data).__name__}")
