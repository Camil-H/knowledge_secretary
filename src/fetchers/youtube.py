"""YouTube uploads-feed + transcript fetching. Stateless."""

import logging
import re

from youtube_transcript_api import YouTubeTranscriptApi

from src.fetchers import rss

logger = logging.getLogger(__name__)

_UPLOADS_FEED = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/)([\w-]{11})")


def channel_videos(channel_id: str) -> dict:
    """Recent uploads via the channel's videos.xml feed.

    Returns {"channel": <title>, "videos": [...]} where each video is
    {video_id, title, url, published (tz-aware UTC | None), summary}.
    """
    feed = rss.fetch(_UPLOADS_FEED.format(channel_id=channel_id))
    videos = []
    for e in feed["entries"]:
        video_id = e["raw"].get("yt_videoid")
        if not video_id:
            continue
        videos.append(
            {
                "video_id": video_id,
                "title": e["title"],
                "url": e["link"] or f"https://www.youtube.com/watch?v={video_id}",
                "published": e["published"],
                "summary": e["summary"],
            }
        )
    return {"channel": feed["title"], "videos": videos}


def transcript(video_id: str) -> str:
    """Best-effort transcript text (any language); empty string on failure."""
    try:
        return _fetch_transcript_text(video_id)
    except Exception as e:
        logger.warning("⚠️ youtube transcript %s failed: %s", video_id, e)
        return ""


def video_id_from_url(url: str) -> str | None:
    m = _VIDEO_ID_RE.search(url or "")
    return m.group(1) if m else None


# == Helper Functions =========================================================


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
    return " ".join(_segment_text(s) for s in segments)


def _segment_text(seg) -> str:
    # youtube-transcript-api returns dicts (<=0.6) or snippet objects (>=1.0)
    return seg["text"] if isinstance(seg, dict) else getattr(seg, "text", "")
