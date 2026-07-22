"""YouTube source adapter + enricher — thin mappers over src/fetchers.youtube.
Kind: yt_channel. Enricher: transcript (channel-id resolution cached in state)."""

from datetime import datetime

from src.core import state as state_mod
from src.core.models import Item
from src.core.registry import enrichers, sources
from src.fetchers import youtube as yt

_KV_PREFIX = "yt_channel:"


# == Source adapter ===========================================================


@sources.register("yt_channel")
def yt_channel(spec: dict, since: datetime, state: dict) -> list[Item]:
    """A YouTube channel's uploads feed, resolved from spec['handle'] (cached)."""
    channel_id = _channel_id(spec, state)
    if channel_id is None:
        return []
    data = yt.channel_videos(channel_id)
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


# == Helper Functions =========================================================


def _channel_id(spec: dict, state: dict) -> str | None:
    """Resolve spec['handle'] -> channel_id, caching the result in state KV."""
    handle = spec["handle"]
    key = _KV_PREFIX + handle
    channel_id = state_mod.get_kv(state, key)
    if not channel_id:
        channel_id = yt.resolve_channel_id(handle)
        if channel_id:
            state_mod.set_kv(state, key, channel_id)
    return channel_id
