"""RSS/Atom feed fetching. Deterministic; degrades to an empty feed on failure."""

import calendar
import logging
from datetime import UTC, datetime

import feedparser
import httpx

from src import config
from src.core.url_guard import assert_safe_url

logger = logging.getLogger(__name__)


def fetch(url: str) -> dict:
    """Fetch + parse a feed.

    Returns {"title": <feed title>, "entries": [...]} where each entry is
    {id, title, link, published (tz-aware UTC | None), summary, raw}. `raw` is the
    underlying feedparser entry, for callers that need extras (e.g. yt_videoid).
    """
    try:
        assert_safe_url(url)
        resp = httpx.get(url, timeout=config.HTTP_TIMEOUT_S)
        parsed = feedparser.parse(resp.content)
        status = resp.status_code
        if status >= 400 or (getattr(parsed, "bozo", 0) and not parsed.entries):
            exc = getattr(parsed, "bozo_exception", "")
            logger.warning("⚠️ rss %s degraded: status=%s %s", url, status, exc)
        entries = [
            {
                "id": e.get("id") or e.get("link", ""),
                "title": e.get("title", ""),
                "link": e.get("link", ""),
                "published": _published_utc(e),
                "summary": e.get("summary", ""),
                "raw": e,
            }
            for e in parsed.entries
        ]
        return {"title": parsed.feed.get("title", ""), "entries": entries}
    except Exception as e:  # guard rejection, httpx transport, or a malformed feed
        logger.warning("⚠️ rss %s degraded: %s", url, e)
        return {"title": "", "entries": []}


# == Helper Functions =========================================================


def _published_utc(entry) -> datetime | None:
    """feedparser's *_parsed struct_time -> tz-aware UTC datetime."""
    struct_time = entry.get("published_parsed") or entry.get("updated_parsed")
    if struct_time is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(struct_time), tz=UTC)
