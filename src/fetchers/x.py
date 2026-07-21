"""X/Twitter fetching via the `twitter-cli` tool (github.com/public-clis/twitter-cli).

Auth is read from the TWITTER_AUTH_TOKEN + TWITTER_CT0 env vars.
"""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

_CLI = "twitter"
_LIST_KEYS = ("tweets", "data", "results")
_DEFAULT_LIMIT = 20


def recent_tweets(handle: str, *, limit: int = _DEFAULT_LIMIT) -> list[dict]:
    """Recent tweets for a handle via `twitter user-posts <handle> --max N --json`.

    Best-effort: needs X auth (TWITTER_AUTH_TOKEN + TWITTER_CT0) and degrades to []
    on any failure. Each tweet dict carries whatever the CLI emits; `_tweet_item`
    reads id/text/url/created_at tolerantly.

    #NOTE: subcommand/flags match the twitter-cli README; exact JSON field names
    are per its SCHEMA.md — adjust `_extract`/`_tweet_item` if a live run differs.
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
        return []
    return data if isinstance(data, list) else []
