"""openRxiv (bioRxiv + medRxiv) recent-preprint fetching via the details API. Degrades to []."""

import logging
from datetime import UTC, datetime

import httpx

from src import config
from src.core.url_guard import UnsafeURLError, assert_safe_url

logger = logging.getLogger(__name__)

_API = "https://api.biorxiv.org/details/{server}/{frm}/{to}/{cursor}/json"


def recent(server: str, categories: list[str], since: datetime) -> list[dict]:
    """Recent `server` ("biorxiv"|"medrxiv") preprints in `categories` (case-insensitive).

    Each: {doi, title, abstract, published (tz-aware UTC), category}. A failed page returns
    whatever was collected so far rather than dropping the batch.
    """
    today = datetime.now(UTC)
    wanted = {c.lower() for c in categories}
    out: list[dict] = []
    cursor = 0
    for _ in range(config.OPENRXIV_MAX_PAGES):
        url = _API.format(
            server=server, frm=f"{since:%Y-%m-%d}", to=f"{today:%Y-%m-%d}", cursor=cursor
        )
        try:
            assert_safe_url(url)
            payload = httpx.get(url, timeout=config.HTTP_TIMEOUT_S).json()
        except (UnsafeURLError, httpx.HTTPError, ValueError) as e:  # rejected, unreachable, or bad
            logger.warning("⚠️ %s degraded: %s", server, e)
            break
        batch = payload.get("collection") or []
        out.extend(_record(e, since) for e in batch if _matches(e, wanted))
        cursor += len(batch)
        total = int((payload.get("messages") or [{}])[0].get("total") or 0)
        if not batch or cursor >= total:
            break
    else:
        logger.warning(
            "⚠️ %s: hit page cap %d; window may be truncated", server, config.OPENRXIV_MAX_PAGES
        )
    return out


# == Helper Functions =========================================================


def _matches(entry: dict, wanted: set[str]) -> bool:
    return entry.get("category", "").lower() in wanted and bool(
        entry.get("doi") and entry.get("date")
    )


def _record(entry: dict, since: datetime) -> dict:
    return {
        "doi": entry["doi"],
        "title": entry.get("title", ""),
        "abstract": entry.get("abstract", ""),
        "published": _parse_date(entry["date"], since),
        "category": entry.get("category", ""),
    }


def _parse_date(raw: str, fallback: datetime) -> datetime:
    """Best-effort openRxiv date ("2024-03-01") -> UTC; malformed rows fall back to `fallback`."""
    try:
        return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return fallback
