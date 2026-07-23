"""bioRxiv recent-preprint fetching via the details API. Degrades to []."""

import logging
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)

_API = "https://api.biorxiv.org/details/biorxiv/{frm}/{to}/0"
_HTTP_TIMEOUT_S = 30


def recent(categories: list[str], since: datetime) -> list[dict]:
    """Recent preprints in `categories` (case-insensitive).

    Each: {doi, title, abstract, published (tz-aware UTC), category}.
    """
    try:
        today = datetime.now(UTC)
        url = _API.format(frm=f"{since:%Y-%m-%d}", to=f"{today:%Y-%m-%d}")
        wanted = {c.lower() for c in categories}
        out = []
        for e in httpx.get(url, timeout=_HTTP_TIMEOUT_S).json().get("collection", []):
            if e.get("category", "").lower() not in wanted:
                continue
            doi, date_str = e.get("doi"), e.get("date")
            if not doi or not date_str:
                continue
            out.append(
                {
                    "doi": doi,
                    "title": e.get("title", ""),
                    "abstract": e.get("abstract", ""),
                    "published": datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC),
                    "category": e.get("category", ""),
                }
            )
        return out
    except (httpx.HTTPError, ValueError) as e:  # bioRxiv unreachable or unparseable response
        logger.warning("⚠️ biorxiv degraded: %s", e)
        return []
