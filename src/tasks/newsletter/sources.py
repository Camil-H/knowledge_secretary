"""Newsletter source adapters + enrichers.

Kinds: `feed` (plain RSS/Atom), `pubmed`, `biorxiv`, `twitter`. Enricher:
`article_text`. Every adapter/enricher degrades gracefully — a single source
failing is logged once here and swallowed, never raised. See CONTRACTS.md.
"""

import calendar
import json
import logging
import subprocess
from datetime import UTC, datetime

import feedparser
import httpx
import trafilatura

from src.core.models import Item
from src.core.registry import enrichers, sources

logger = logging.getLogger(__name__)

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_PUBMED_DATE_FORMATS = ("%Y %b %d", "%Y %b", "%Y/%m/%d", "%Y-%m-%d", "%Y")
_HTTP_TIMEOUT_S = 30
_PUBMED_RETMAX = 30
_TWEET_LIMIT = 20
# agent-reach installs the `twitter` (twitter-cli) backend; a user timeline is a
# `from:` search. Documented usage: `twitter search "query" -n 10`.
_TWITTER_CLI = "twitter"
_TWEET_LIST_KEYS = ("tweets", "data", "results")


# == Source adapters ==========================================================


@sources.register("feed")
def feed(spec: dict, since: datetime, state: dict) -> list[Item]:
    """Plain RSS/Atom feed (blogs, news sites, journal TOCs, agency feeds)."""
    try:
        parsed = feedparser.parse(spec["url"])
        items = []
        for entry in parsed.entries:
            published = _struct_to_utc(entry.get("published_parsed") or entry.get("updated_parsed"))
            if published is None:
                continue
            items.append(
                Item(
                    id="rss:" + (entry.get("id") or entry.link),
                    source=spec["key"],
                    section=spec["section"],
                    title=entry.title,
                    url=entry.link,
                    published=published,
                    text=entry.get("summary", ""),
                )
            )
        return items
    except Exception as e:
        logger.warning("⚠️ feed %s failed: %s", spec.get("key"), e)
        return []


@sources.register("pubmed")
def pubmed(spec: dict, since: datetime, state: dict) -> list[Item]:
    """Recent PubMed hits for spec['queries'] via NCBI E-utilities (esearch+esummary)."""
    try:
        reldate = max(1, (datetime.now(UTC) - since).days)
        idlist = (
            httpx.get(
                f"{_EUTILS}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": " OR ".join(spec["queries"]),
                    "datetype": "pdat",
                    "reldate": reldate,
                    "retmax": _PUBMED_RETMAX,
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

        items = []
        for pmid in result.get("uids", idlist):
            doc = result.get(pmid, {})
            title = doc.get("title", "")
            if not title:
                continue
            items.append(
                Item(
                    id="pubmed:" + pmid,
                    source=spec["key"],
                    section=spec["section"],
                    title=title,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    published=_parse_pubmed_date(doc.get("pubdate", ""), since),
                    text=title,
                )
            )
        return items
    except Exception as e:
        logger.warning("⚠️ pubmed %s failed: %s", spec.get("key"), e)
        return []


@sources.register("biorxiv")
def biorxiv(spec: dict, since: datetime, state: dict) -> list[Item]:
    """Recent bioRxiv preprints in spec['categories'] via the bioRxiv details API."""
    try:
        today = datetime.now(UTC)
        url = f"https://api.biorxiv.org/details/biorxiv/{since:%Y-%m-%d}/{today:%Y-%m-%d}/0"
        wanted = {c.lower() for c in spec["categories"]}
        items = []
        for entry in httpx.get(url, timeout=_HTTP_TIMEOUT_S).json().get("collection", []):
            if entry.get("category", "").lower() not in wanted:
                continue
            doi, date_str = entry.get("doi"), entry.get("date")
            if not doi or not date_str:
                continue
            items.append(
                Item(
                    id="biorxiv:" + doi,
                    source=spec["key"],
                    section=spec["section"],
                    title=entry.get("title", ""),
                    url="https://doi.org/" + doi,
                    published=datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC),
                    text=entry.get("abstract", ""),
                )
            )
        return items
    except Exception as e:
        logger.warning("⚠️ biorxiv %s failed: %s", spec.get("key"), e)
        return []


@sources.register("twitter")
def twitter(spec: dict, since: datetime, state: dict) -> list[Item]:
    """Best-effort recent tweets via the agent-reach `twitter` backend, per handle.

    Runs `twitter search "from:<handle>" -n N --json`. The backend is frequently
    unavailable (stale X cookie, not installed), so each handle is isolated and any
    failure just yields no items for that handle.

    #TODO: binary/subcommand (`twitter search`, `-n`) are from agent-reach's docs,
    but the JSON output flag/shape is UNVERIFIED — confirm with one live run and
    adjust `--json` / `_load_tweets` if the tool emits a different structure.
    """
    items: list[Item] = []
    for handle in spec.get("handles", []):
        try:
            proc = subprocess.run(
                [_TWITTER_CLI, "search", f"from:{handle}", "-n", str(_TWEET_LIMIT), "--json"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            parsed = (_tweet_to_item(t, spec, handle) for t in _load_tweets(proc.stdout))
            items.extend(it for it in parsed if it is not None)
        except Exception as e:
            logger.warning("⚠️ twitter %s degraded (no items): %s", handle, e)
    return items


# == Enrichers ================================================================


@enrichers.register("article_text")
def article_text(item: Item) -> Item:
    """Replace item.text with the extracted main article body (trafilatura)."""
    try:
        downloaded = trafilatura.fetch_url(item.url)
        extracted = trafilatura.extract(downloaded) if downloaded else None
        if extracted:
            item.text = extracted
    except Exception as e:
        logger.warning("⚠️ article_text %s failed: %s", item.id, e)
    return item


# == Helper Functions =========================================================


def _struct_to_utc(struct_time) -> datetime | None:
    """feedparser's *_parsed struct_time -> tz-aware UTC datetime."""
    if struct_time is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(struct_time), tz=UTC)


def _parse_pubmed_date(raw: str, fallback: datetime) -> datetime:
    """Best-effort PubMed pubdate ("2024 Jan 15" / "2024 Jan" / "2024") -> UTC."""
    raw = (raw or "").split(" (")[0].strip()
    for fmt in _PUBMED_DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return fallback


def _load_tweets(stdout: str) -> list[dict]:
    """Tolerate either a top-level JSON array or an object wrapping the tweet list."""
    data = json.loads(stdout)
    if isinstance(data, dict):
        for key in _TWEET_LIST_KEYS:
            if isinstance(data.get(key), list):
                return data[key]
        return []
    return data if isinstance(data, list) else []


def _tweet_to_item(t: dict, spec: dict, handle: str) -> Item | None:
    tweet_id = str(t.get("id") or t.get("tweet_id") or "")
    if not tweet_id:
        return None
    try:
        published = datetime.fromisoformat(
            (t.get("created_at") or t.get("date") or "").replace("Z", "+00:00")
        ).astimezone(UTC)
    except ValueError:
        return None
    text = t.get("text", "")
    return Item(
        id="x:" + tweet_id,
        source=spec["key"],
        section=spec["section"],
        title=text[:80],
        url=t.get("url") or f"https://x.com/{handle}/status/{tweet_id}",
        published=published,
        text=text,
        meta={"handle": handle},
    )
