"""YouTube source adapter + enricher — thin mappers over src/fetchers.youtube.
Kind: yt_channel (by exact channel_id). Enricher: transcript."""

from datetime import datetime

from src.core.models import Item, SourceSpec, State
from src.core.registry import enrichers, sources
from src.fetchers import youtube as yt

# == Source adapter ===========================================================


@sources.register("yt_channel")
def yt_channel(spec: SourceSpec, since: datetime, state: State) -> list[Item]:
    """A YouTube channel's uploads feed, keyed by the exact spec['channel_id']."""
    data = yt.channel_videos(spec["channel_id"])
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
                meta={"channel": data["channel"]},
            )
        )
    return items


# == Enrichers ================================================================


@enrichers.register("transcript")
def transcript(item: Item) -> Item:
    """Prefer the video's transcript (any language); degrade to the RSS description
    already in item.text when the transcript is unavailable — YouTube blocks the CI
    runner's datacenter IP, so an empty transcript is the common case, not the error
    case, and must not clobber the description we already have. The chosen source is
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
