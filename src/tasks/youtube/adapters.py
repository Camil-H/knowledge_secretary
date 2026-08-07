"""YouTube source adapter + enricher — thin mappers over src/fetchers.youtube.
Kind: yt_channel (by exact channel_id). Enricher: transcript."""

from datetime import datetime

from src.core.models import Item, SourceSpec
from src.core.registry import enrichers, sources
from src.fetchers import youtube as yt

WATCH_MODE = "watch"

# == Source adapter ===========================================================


@sources.register("yt_channel")
def yt_channel(spec: SourceSpec, since: datetime) -> list[Item]:
    """A YouTube channel's uploads feed, keyed by the exact spec['channel_id'].

    `mode: watch` marks a channel watch-only: its videos carry meta['watch'] and the task
    lists them unsummarized. Such a spec also omits `enrich`, so no transcript is fetched
    for a video nobody is going to summarize."""
    data = yt.channel_videos(spec["channel_id"])
    watch = spec.get("mode") == WATCH_MODE
    items = []
    for v in data["videos"]:
        if v["published"] is None:
            continue
        items.append(
            Item(
                id="yt:" + v["video_id"],
                source=spec["key"],
                section=spec["section"],
                title=v["title"],
                url=v["url"],
                published=v["published"],
                text=v["summary"],
                meta={"channel": data["channel"], "watch": watch},
            )
        )
    return items


# == Enrichers ================================================================


@enrichers.register("transcript")
def transcript(item: Item) -> Item:
    """Prefer the video's transcript (any language); degrade to the RSS description
    already in item.text when the transcript is unavailable. The chosen source is
    recorded in item.meta['text_source'] so downstream can flag lower-confidence items."""
    video_id = (
        item.id[len("yt:") :] if item.id.startswith("yt:") else yt.video_id_from_url(item.url)
    )
    fetched = yt.transcript(video_id) if video_id else ""
    if fetched.strip():
        item.text = fetched
        item.meta["text_source"] = "transcript"
    else:
        item.meta["text_source"] = "description" if item.text.strip() else "title"
    return item
