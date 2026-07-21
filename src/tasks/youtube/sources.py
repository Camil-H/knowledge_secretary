"""YouTube source adapter + enricher.

Kind: `yt_channel` (a channel's uploads feed, resolved from its @handle and
cached in state). Enricher: `transcript`. Both degrade gracefully — failures are
logged once here and swallowed, never raised. See CONTRACTS.md.
"""

import calendar
import logging
import re
from datetime import UTC, datetime

import feedparser
import httpx
from youtube_transcript_api import YouTubeTranscriptApi

from src.core import state as state_mod
from src.core.models import Item
from src.core.registry import enrichers, sources

logger = logging.getLogger(__name__)

_YT_FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
_YT_CHANNEL_ID_RE = re.compile(r'"channelId":"(UC[\w-]+)"')
_YT_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/)([\w-]{11})")
_HTTP_TIMEOUT_S = 30


# == Source adapter ===========================================================


@sources.register("yt_channel")
def yt_channel(spec: dict, since: datetime, state: dict) -> list[Item]:
    """A YouTube channel's uploads feed, resolved from spec['handle'] (cached)."""
    try:
        channel_id = _resolve_channel_id(spec, state)
        if channel_id is None:
            return []
        parsed = feedparser.parse(_YT_FEED_URL.format(channel_id=channel_id))
        items = []
        for entry in parsed.entries:
            published = _struct_to_utc(entry.get("published_parsed") or entry.get("updated_parsed"))
            if published is None or not hasattr(entry, "yt_videoid"):
                continue
            items.append(
                Item(
                    id="yt:" + entry.yt_videoid,
                    source=spec["key"],
                    section=spec["section"],
                    title=entry.title,
                    url=entry.get("link", f"https://www.youtube.com/watch?v={entry.yt_videoid}"),
                    published=published,
                    text=entry.get("summary", ""),
                    meta={"channel": parsed.feed.get("title")},
                )
            )
        return items
    except Exception as e:
        logger.warning("⚠️ yt_channel %s failed: %s", spec.get("key"), e)
        return []


# == Enrichers ================================================================


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


# == Helper Functions =========================================================


def _resolve_channel_id(spec: dict, state: dict) -> str | None:
    """Resolve + cache a channel_id from spec['handle'] (scrape the channel page)."""
    handle = spec["handle"]
    kv_key = f"yt_channel:{handle}"
    channel_id = state_mod.get_kv(state, kv_key)
    if not channel_id:
        resp = httpx.get(
            f"https://www.youtube.com/{handle}", timeout=_HTTP_TIMEOUT_S, follow_redirects=True
        )
        m = _YT_CHANNEL_ID_RE.search(resp.text)
        if not m:
            logger.warning(
                "⚠️ yt_channel %s: could not resolve channelId for %s", spec.get("key"), handle
            )
            return None
        channel_id = m.group(1)
        state_mod.set_kv(state, kv_key, channel_id)
    return channel_id


def _struct_to_utc(struct_time) -> datetime | None:
    """feedparser's *_parsed struct_time -> tz-aware UTC datetime."""
    if struct_time is None:
        return None
    return datetime.fromtimestamp(calendar.timegm(struct_time), tz=UTC)


def _yt_id_from_url(url: str) -> str | None:
    m = _YT_VIDEO_ID_RE.search(url or "")
    return m.group(1) if m else None


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
