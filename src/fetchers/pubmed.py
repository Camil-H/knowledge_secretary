"""PubMed recent-article search via NCBI E-utilities. Degrades to [] on failure."""

import logging
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_DATE_FORMATS = ("%Y %b %d", "%Y %b", "%Y/%m/%d", "%Y-%m-%d", "%Y")
_HTTP_TIMEOUT_S = 30
_DEFAULT_RETMAX = 30


def search_recent(
    queries: list[str], since: datetime, *, retmax: int = _DEFAULT_RETMAX
) -> list[dict]:
    """Recent PubMed hits (esearch + esummary). Each: {pmid, title, published (UTC)}."""
    try:
        reldate = max(1, (datetime.now(UTC) - since).days)
        idlist = (
            httpx.get(
                f"{_EUTILS}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": " OR ".join(queries),
                    "datetype": "pdat",
                    "reldate": reldate,
                    "retmax": retmax,
                    "sort": "date",
                    "retmode": "json",
                },
                timeout=_HTTP_TIMEOUT_S,
            )
            .json()
            .get("esearchresult", {})
            .get("idlist", [])
        )
        if not idlist:
            return []

        result = (
            httpx.get(
                f"{_EUTILS}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(idlist), "retmode": "json"},
                timeout=_HTTP_TIMEOUT_S,
            )
            .json()
            .get("result", {})
        )
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
    except (httpx.HTTPError, ValueError) as e:  # NCBI unreachable or unparseable response
        logger.warning("⚠️ pubmed degraded: %s", e)
        return []


# == Helper Functions =========================================================


def _parse_date(raw: str, fallback: datetime) -> datetime:
    """Best-effort PubMed pubdate ("2024 Jan 15" / "2024 Jan" / "2024") -> UTC."""
    raw = (raw or "").split(" (")[0].strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return fallback
