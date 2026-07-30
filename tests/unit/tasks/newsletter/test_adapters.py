"""Newsletter source adapters + enrichers. Fetcher collaborators are stubbed."""

import threading
from datetime import UTC, datetime

import pytest

from src.core.errors import AuthError
from src.core.models import Item
from src.fetchers import x
from src.tasks.newsletter import adapters
from src.tasks.newsletter.adapters import (
    _tweet_item,
    article_text,
    biorxiv_source,
    feed,
    medrxiv_source,
    pubmed_source,
    twitter,
)

_SINCE = datetime(2024, 1, 1, tzinfo=UTC)


def _spec(**extra):
    return {"key": "src1", "section": "News", **extra}


def _item(*, url="http://u", text="orig"):
    return Item(
        id="i1",
        source="src1",
        section="News",
        title="T",
        url=url,
        published=_SINCE,
        text=text,
    )


# == Source adapters ==========================================================

# ----- feed -----


def test_feed_maps_entries_and_skips_unpublished(monkeypatch):
    entries = [
        {"id": "e1", "title": "T1", "link": "http://l1", "published": _SINCE, "summary": "s1"},
        {"id": "e2", "title": "T2", "link": "http://l2", "published": None, "summary": "s2"},
    ]
    monkeypatch.setattr(adapters.rss, "fetch", lambda _url: {"entries": entries})

    out = feed(_spec(url="http://feed"), _SINCE)

    assert [i.id for i in out] == ["rss:e1"]  # published-None entry skipped
    item = out[0]
    assert (item.source, item.section) == ("src1", "News")
    assert (item.title, item.url, item.published, item.text) == ("T1", "http://l1", _SINCE, "s1")


# ----- pubmed_source -----


def test_pubmed_source_maps_and_forwards_queries_since(monkeypatch):
    seen = {}

    def _search_recent(queries, since):
        seen["queries"], seen["since"] = queries, since
        return [{"pmid": "123", "title": "Ti", "published": _SINCE}]

    monkeypatch.setattr(adapters.pubmed, "search_recent", _search_recent)

    out = pubmed_source(_spec(queries=["q1", "q2"]), _SINCE)

    assert seen == {"queries": ["q1", "q2"], "since": _SINCE}
    item = out[0]
    assert item.id == "pubmed:123"
    assert item.url == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert (item.source, item.section, item.title, item.text) == ("src1", "News", "Ti", "Ti")


# ----- biorxiv_source / medrxiv_source -----


@pytest.mark.parametrize(
    ("source_fn", "server", "doi", "category"),
    [
        (biorxiv_source, "biorxiv", "10.1/abc", "cs.AI"),
        (medrxiv_source, "medrxiv", "10.2/xyz", "oncology"),
    ],
    ids=["biorxiv", "medrxiv"],
)
def test_preprint_source_maps_records_and_routes_to_its_server(
    monkeypatch, source_fn, server, doi, category
):
    seen = {}

    def _recent(requested_server, categories, since):
        seen["server"], seen["categories"], seen["since"] = requested_server, categories, since
        return [{"doi": doi, "title": "Ti", "abstract": "Abs", "published": _SINCE}]

    monkeypatch.setattr(adapters.openrxiv, "recent", _recent)

    out = source_fn(_spec(categories=[category]), _SINCE)

    assert seen == {"server": server, "categories": [category], "since": _SINCE}
    item = out[0]
    assert item.id == f"{server}:{doi}"
    assert item.url == "https://doi.org/" + doi
    assert (item.source, item.section, item.title, item.text) == ("src1", "News", "Ti", "Abs")
    assert item.published == _SINCE


# ----- twitter -----


def _unreachable_fetch(handle: str) -> list[dict]:
    raise AssertionError(f"no handle should have been fetched, got {handle}")


def _tweets_for(handle: str) -> list[dict]:
    return [{"id": f"{handle}1", "createdAtISO": "2024-01-01T00:00:00Z", "text": "hi"}]


def test_twitter_fetches_handles_concurrently_keeping_handle_order(monkeypatch):
    signal = threading.Event()
    calls = []

    def _recent_tweets(handle):
        calls.append(handle)
        # h1 can only finish once h2 has run, so a sequential walk would time out here
        if handle == "h1":
            if not signal.wait(timeout=5):
                raise TimeoutError("handle fetches never overlapped")
        else:
            signal.set()
        return _tweets_for(handle)

    monkeypatch.setattr(adapters.x, "recent_tweets", _recent_tweets)

    out = twitter(_spec(handles=["h1", "h2"]), _SINCE)

    assert sorted(calls) == ["h1", "h2"]  # exactly one subprocess per configured handle
    assert [i.id for i in out] == ["x:h11", "x:h21"]
    # each Item is attributed to the handle it was fetched for, permalinks included
    assert [i.meta["handle"] for i in out] == ["h1", "h2"]
    assert [i.url for i in out] == ["https://x.com/h1/status/h11", "https://x.com/h2/status/h21"]


def test_twitter_handle_that_degraded_to_empty_leaves_the_others(monkeypatch):
    monkeypatch.setattr(
        adapters.x, "recent_tweets", lambda handle: [] if handle == "dead" else _tweets_for(handle)
    )

    out = twitter(_spec(handles=["dead", "alive"]), _SINCE)

    assert [i.id for i in out] == ["x:alive1"]


def test_twitter_without_handles_fetches_nothing(monkeypatch):
    monkeypatch.setattr(adapters.x, "recent_tweets", _unreachable_fetch)

    assert twitter(_spec(), _SINCE) == []


def test_twitter_propagates_an_auth_failure_from_any_handle(monkeypatch):
    boom = AuthError("x", detail="creds expired")

    def _recent_tweets(handle):
        if handle == "h2":
            raise boom
        return _tweets_for(handle)

    monkeypatch.setattr(adapters.x, "recent_tweets", _recent_tweets)

    with pytest.raises(AuthError) as ei:
        twitter(_spec(handles=["h1", "h2"]), _SINCE)

    assert ei.value is boom


# == Enrichers ================================================================

# ----- article_text -----


@pytest.mark.parametrize(
    ("body", "expected"),
    [("full body", "full body"), (None, "orig"), ("", "orig")],
    ids=["present", "none", "empty"],
)
def test_article_text_replaces_the_body_only_when_extraction_yields_text(
    monkeypatch, body, expected
):
    monkeypatch.setattr(adapters.url, "article_text", lambda _u: body)

    out = article_text(_item(text="orig"))

    assert out.text == expected


# == Helper Functions =========================================================

# ----- _tweet_item -----


@pytest.mark.parametrize(
    "tweet",
    [
        {"createdAtISO": "2024-01-01T00:00:00Z"},
        {"id": "1"},
    ],
    ids=["missing_id", "missing_date"],
)
def test_tweet_item_missing_id_or_date_raises(tweet):
    with pytest.raises(x.UnexpectedXFormat):
        _tweet_item(tweet, _spec(), "h")


def test_tweet_item_unparseable_date_chains_value_error():
    tweet = {"id": "1", "createdAtISO": "not-a-date"}

    with pytest.raises(x.UnexpectedXFormat) as ei:
        _tweet_item(tweet, _spec(), "h")

    assert isinstance(ei.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("date_key", "raw_date"),
    [
        ("createdAtISO", "2024-06-01T12:00:00+02:00"),
        ("createdAt", "2024-06-01T10:00:00Z"),
    ],
    ids=["iso_key", "createdat_fallback"],
)
def test_tweet_item_builds_the_item_from_a_well_formed_tweet(date_key, raw_date):
    """Either date key lands on the same UTC instant: the offset-bearing row pins the
    astimezone(UTC) conversion, the Z-suffixed row the shape X actually returns. Dropping
    either key's lookup leaves its row dateless, so precedence stays pinned both ways."""
    text = "x" * 100
    tweet = {"id": "42", date_key: raw_date, "text": text}

    item = _tweet_item(tweet, _spec(), "handle1")

    assert item.id == "x:42"
    # isoformat, not equality: aware datetimes compare by instant, so == would not see a
    # published still carrying the +02:00 offset instead of the normalized UTC one
    assert item.published.isoformat() == "2024-06-01T10:00:00+00:00"
    assert item.url == "https://x.com/handle1/status/42"  # no tweet url -> permalink fallback
    assert item.title == text[:80]
    assert len(item.title) == 80
    assert (item.source, item.section, item.text) == ("src1", "News", text)
    assert item.meta == {"handle": "handle1"}
