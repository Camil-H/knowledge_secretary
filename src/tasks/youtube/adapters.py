"""YouTube source adapter + enricher — thin mappers over src/fetchers.youtube.
Kind: yt_channel (by exact channel_id). Enricher: transcript."""

from datetime import datetime

from src.core.models import Item
from src.core.registry import enrichers, sources
from src.fetchers import youtube as yt

# == Source adapter ===========================================================


@sources.register("yt_channel")
def yt_channel(spec: dict, since: datetime, state: dict) -> list[Item]:
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
    """Set item.text to the video's transcript (any available language)."""
    video_id = (
        item.id[len("yt:") :] if item.id.startswith("yt:") else yt.video_id_from_url(item.url)
    )
    if video_id:
        item.text = yt.transcript(video_id)
    return item
