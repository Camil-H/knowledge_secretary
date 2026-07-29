"""PubMed recent-article search via NCBI E-utilities. Degrades to [] on failure."""

import logging
from datetime import UTC, datetime

import httpx

from src import config
from src.core.url_guard import UnsafeURLError, assert_safe_url

logger = logging.getLogger(__name__)

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_DATE_FORMATS = ("%Y %b %d", "%Y %b", "%Y/%m/%d", "%Y-%m-%d", "%Y")


def search_recent(
    queries: list[str], since: datetime, *, retmax: int = config.PUBMED_RETMAX
) -> list[dict]:
    """Recent PubMed hits (esearch + esummary). Each: {pmid, title, published (UTC)}."""
    try:
        reldate = max(1, (datetime.now(UTC) - since).days)
        idlist = (
            _guarded_json(
                f"{_EUTILS}/esearch.fcgi",
                {
                    "db": "pubmed",
                    "term": " OR ".join(queries),
                    "datetype": "pdat",
                    "reldate": reldate,
                    "retmax": retmax,
                    "sort": "date",
                    "retmode": "json",
                },
            )
            .get("esearchresult", {})
            .get("idlist", [])
        )
        if not idlist:
            return []

        result = _guarded_json(
            f"{_EUTILS}/esummary.fcgi",
            {"db": "pubmed", "id": ",".join(idlist), "retmode": "json"},
        ).get("result", {})
        out = []
        for pmid in result.get("uids", idlist):
            title = result.get(pmid, {}).get("title", "")
            if not title:
                continue
            out.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "published": _parse_date(result[pmid].get("pubdate", ""), since),
                }
            )
        return out
    except (UnsafeURLError, httpx.HTTPError, ValueError) as e:  # rejected, unreachable, unparseable
        logger.warning("⚠️ pubmed degraded: %s", e)
        return []


# == Helper Functions =========================================================


def _guarded_json(url: str, params: dict[str, str | int]) -> dict:
    """GET an E-utilities endpoint through the SSRF guard and return its parsed JSON."""
    assert_safe_url(url)
    return httpx.get(url, params=params, timeout=config.HTTP_TIMEOUT_S).json()


def _parse_date(raw: str, fallback: datetime) -> datetime:
    """Best-effort PubMed pubdate ("2024 Jan 15" / "2024 Jan" / "2024") -> UTC."""
    raw = (raw or "").split(" (")[0].strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return fallback
