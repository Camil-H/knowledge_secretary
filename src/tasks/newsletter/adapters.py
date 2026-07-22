"""Newsletter source adapters + enrichers — thin mappers over src/fetchers.
Kinds: feed/pubmed/biorxiv/twitter. Enricher: article_text."""

from datetime import UTC, datetime

from src.core.models import Item
from src.core.registry import enrichers, sources
from src.fetchers import biorxiv, pubmed, rss, url, x

# == Source adapters ==========================================================


@sources.register("feed")
def feed(spec: dict, since: datetime, state: dict) -> list[Item]:
    """Plain RSS/Atom feed (blogs, news sites, journal TOCs, agency feeds)."""
    items = []
    for e in rss.fetch(spec["url"])["entries"]:
        if e["published"] is None:
            continue
        items.append(
            Item(
                id="rss:" + e["id"],
                source=spec["key"],
                section=spec["section"],
                title=e["title"],
                url=e["link"],
                published=e["published"],
                text=e["summary"],
            )
        )
    return items


@sources.register("pubmed")
def pubmed_source(spec: dict, since: datetime, state: dict) -> list[Item]:
    return [
        Item(
            id="pubmed:" + r["pmid"],
            source=spec["key"],
            section=spec["section"],
            title=r["title"],
            url=f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/",
            published=r["published"],
            text=r["title"],
        )
        for r in pubmed.search_recent(spec["queries"], since)
    ]


@sources.register("biorxiv")
def biorxiv_source(spec: dict, since: datetime, state: dict) -> list[Item]:
    return [
        Item(
            id="biorxiv:" + r["doi"],
            source=spec["key"],
            section=spec["section"],
            title=r["title"],
            url="https://doi.org/" + r["doi"],
            published=r["published"],
            text=r["abstract"],
        )
        for r in biorxiv.recent(spec["categories"], since)
    ]


@sources.register("twitter")
def twitter(spec: dict, since: datetime, state: dict) -> list[Item]:
    items = []
    for handle in spec.get("handles", []):
        for tweet in x.recent_tweets(handle):
            item = _tweet_item(tweet, spec, handle)
            if item is not None:
                items.append(item)
    return items


# == Enrichers ================================================================


@enrichers.register("article_text")
def article_text(item: Item) -> Item:
    """Replace item.text with the extracted full article body, if available."""
    body = url.article_text(item.url)
    if body:
        item.text = body
    return item


# == Helper Functions =========================================================


def _tweet_item(tweet: dict, spec: dict, handle: str) -> Item | None:
    tweet_id = str(tweet.get("id") or tweet.get("tweet_id") or "")
    if not tweet_id:
        return None
    try:
        published = datetime.fromisoformat(
            (tweet.get("created_at") or tweet.get("date") or "").replace("Z", "+00:00")
        ).astimezone(UTC)
    except ValueError:
        return None
    text = tweet.get("text", "")
    return Item(
        id="x:" + tweet_id,
        source=spec["key"],
        section=spec["section"],
        title=text[:80],
        url=tweet.get("url") or f"https://x.com/{handle}/status/{tweet_id}",
        published=published,
        text=text,
        meta={"handle": handle},
    )
