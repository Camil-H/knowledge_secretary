"""Source adapters + enrichers, dispatched by `gather`.

Adapters (keyed by `kind`) turn a per-task source spec into list[Item]; enrichers
(keyed by name) post-process an Item (fetch article body, transcript, ...). Every
adapter/enricher degrades gracefully: a single source or enrichment failing must
never take down a run, so failures are logged once here and swallowed (return
[] / return item unchanged) rather than raised.
"""

import calendar
import json
import logging
import re
import subprocess
from datetime import UTC, datetime

import feedparser
import httpx
import trafilatura
from youtube_transcript_api import YouTubeTranscriptApi

from . import state as state_mod
from .models import Item
from .registry import enrichers, sources

logger = logging.getLogger(__name__)

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_YT_FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
_YT_CHANNEL_ID_RE = re.compile(r'"channelId":"(UC[\w-]+)"')
_YT_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/)([\w-]{11})")
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
    """RSS/Atom via feedparser. With a `handle`, resolve+cache a YouTube channel_id
    and read its uploads feed; otherwise read spec['url']."""
    try:
        url = _resolve_feed_url(spec, state)
        if url is None:
            return []
        parsed = feedparser.parse(url)
        items = []
        for entry in parsed.entries:
            published = _struct_to_utc(entry.get("published_parsed") or entry.get("updated_parsed"))
            if published is None:
                continue
            items.append(_entry_to_item(entry, spec, parsed, published))
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


@enrichers.register("transcript")
def transcript(item: Item) -> Item:
    """Set item.text to the video's transcript (any available language)."""
    try:
        video_id = item.id[len("yt:") :] if item.id.startswith("yt:") else _yt_id_from_url(item.url)
        if not video_id:
            return item
        item.text = _fetch_transcript_text(video_id)
    except Exception as e:
        logger.warning("⚠️ transcript %s failed: %s", item.id, e)
    return item


# == gather ===================================================================


def gather(specs: list[dict], state: dict, since: datetime) -> list[Item]:
    """Fetch + dedup + enrich NEW items across a task's own source specs.

    Returns NEW items (state.is_new) published >= since, enriched per spec. Does
    NOT mark items seen — the caller marks only what it actually consumes, so an
    item fetched but dropped downstream (e.g. outside a narrower window) resurfaces
    next run instead of being silently burned.
    """
    gathered: list[Item] = []
    for spec in specs:
        try:
            fetched = sources.get(spec["kind"])(spec, since, state)
        except Exception:
            logger.exception("❌ gather: source %s crashed", spec.get("key"))
            continue
        for item in fetched:
            if not state_mod.is_new(state, item) or item.published < since:
                continue
            for name in spec.get("enrich", []):
                item = enrichers.get(name)(item)
            gathered.append(item)
    return gathered


# == Helper Functions =========================================================

# ----- feed -----


def _resolve_feed_url(spec: dict, state: dict) -> str | None:
    """Feed URL for a spec: cached/resolved YouTube channel feed, or spec['url']."""
    if "handle" not in spec:
        return spec.get("url")
    handle = spec["handle"]
    kv_key = f"yt_channel:{handle}"
    channel_id = state_mod.get_kv(state, kv_key)
    if not channel_id:
        resp = httpx.get(
            f"https://www.youtube.com/{handle}", timeout=_HTTP_TIMEOUT_S, follow_redirects=True
        )
        m = _YT_CHANNEL_ID_RE.search(resp.text)
        if not m:
            logger.warning("⚠️ feed %s: could not resolve channelId for %s", spec.get("key"), handle)
            return None
        channel_id = m.group(1)
        state_mod.set_kv(state, kv_key, channel_id)
    return _YT_FEED_URL.format(channel_id=channel_id)


def _entry_to_item(entry, spec: dict, parsed, published: datetime) -> Item:
    if hasattr(entry, "yt_videoid"):
        return Item(
            id="yt:" + entry.yt_videoid,
            source=spec["key"],
            section=spec["section"],
            title=entry.title,
            url=entry.get("link", f"https://www.youtube.com/watch?v={entry.yt_videoid}"),
            published=published,
            text=entry.get("summary", ""),
            meta={"channel": parsed.feed.get("title")},
        )
    return Item(
        id="rss:" + (entry.get("id") or entry.link),
        source=spec["key"],
        section=spec["section"],
        title=entry.title,
        url=entry.link,
        published=published,
        text=entry.get("summary", ""),
    )


# ----- twitter -----


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


# ----- transcript -----


def _fetch_transcript_text(video_id: str) -> str:
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        langs = [t.language_code for t in listing]
        try:
            tr = listing.find_generated_transcript(langs)
        except Exception:
            tr = listing.find_transcript(langs)
        segments = tr.fetch()
    except Exception:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(_segment_text(seg) for seg in segments)


def _segment_text(seg) -> str:
    # youtube-transcript-api returns dicts (<=0.6) or snippet objects (>=1.0)
    return seg["text"] if isinstance(seg, dict) else getattr(seg, "text", "")


# ----- dates / ids -----


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


def _yt_id_from_url(url: str) -> str | None:
    m = _YT_VIDEO_ID_RE.search(url or "")
    return m.group(1) if m else None
