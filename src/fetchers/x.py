"""X/Twitter fetching via the `twitter-cli` tool; auth from TWITTER_AUTH_TOKEN + TWITTER_CT0."""

import json
import logging
import re
import subprocess

from src.core.errors import AuthError, ExternalError

logger = logging.getLogger(__name__)

_CLI = "twitter"
_LIST_KEYS = ("tweets", "data", "results")
_DEFAULT_LIMIT = 20
# high-signal only — generic words like "session"/"cookie"/"expired" also show up in
# unrelated network/timeout failures and would misclassify them as auth
_AUTH_MARKERS = (
    "unauthorized",
    "forbidden",
    "authenticat",
    "credential",
    "invalid api key",
)
_AUTH_STATUS_RE = re.compile(r"\b(401|403)\b")
# X handles are 1-15 word chars; anchoring the match rules out flag-shaped input
# (e.g. "--json", "-rf") reaching argv as the positional handle.
_HANDLE_RE = re.compile(r"[A-Za-z0-9_]{1,15}")


# == Exceptions ===============================================================


class UnexpectedXFormat(ExternalError):
    """twitter-cli output didn't match the expected tweet schema."""


# == Fetch ====================================================================


def recent_tweets(handle: str, *, limit: int = _DEFAULT_LIMIT) -> list[dict]:
    """Recent tweets via `twitter user-posts <handle> --max N --json`; [] on failure
    (including a malformed handle, so one bad config entry doesn't sink the source)."""
    normalized = handle.removeprefix("@")
    if not _HANDLE_RE.fullmatch(normalized):
        logger.warning("⚠️ x %s degraded: invalid handle format", handle)
        return []
    try:
        proc = subprocess.run(
            [_CLI, "user-posts", normalized, "--max", str(limit), "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return _extract(json.loads(proc.stdout))
    except UnexpectedXFormat:
        raise  # format drift is loud, not a silent degrade
    except subprocess.CalledProcessError as e:
        if _is_auth_failure(e.stderr):  # expired cookies: propagate, don't silently degrade
            raise AuthError(
                "x",
                detail="session cookies rejected — renew TWITTER_AUTH_TOKEN / TWITTER_CT0",
                cause=e,
            ) from e
        logger.warning("⚠️ x %s degraded: %s (exit %s)", handle, type(e).__name__, e.returncode)
        return []
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as e:
        logger.warning("⚠️ x %s degraded: %s", handle, type(e).__name__)
        return []


# == Helper Functions =========================================================


def _is_auth_failure(stderr: str | None) -> bool:
    text = (stderr or "").lower()
    return any(marker in text for marker in _AUTH_MARKERS) or bool(_AUTH_STATUS_RE.search(text))


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
