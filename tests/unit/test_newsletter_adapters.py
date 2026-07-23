"""Newsletter source adapters + enrichers. Fetcher collaborators are stubbed."""

from datetime import UTC, datetime

import pytest

from src.core.models import Item
from src.fetchers import x
from src.tasks.newsletter import adapters
from src.tasks.newsletter.adapters import (
    _tweet_item,
    article_text,
    biorxiv_source,
    feed,
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

    out = feed(_spec(url="http://feed"), _SINCE, {})

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

    out = pubmed_source(_spec(queries=["q1", "q2"]), _SINCE, {})

    assert seen == {"queries": ["q1", "q2"], "since": _SINCE}
    item = out[0]
    assert item.id == "pubmed:123"
    assert item.url == "https://pubmed.ncbi.nlm.nih.gov/123/"
    assert (item.source, item.section, item.title, item.text) == ("src1", "News", "Ti", "Ti")


# ----- biorxiv_source -----


def test_biorxiv_source_maps_doi_url_and_abstract(monkeypatch):
    seen = {}

    def _recent(categories, since):
        seen["categories"], seen["since"] = categories, since
        return [{"doi": "10.1/abc", "title": "Ti", "abstract": "Abs", "published": _SINCE}]

    monkeypatch.setattr(adapters.biorxiv, "recent", _recent)

    out = biorxiv_source(_spec(categories=["cs.AI"]), _SINCE, {})

    assert seen == {"categories": ["cs.AI"], "since": _SINCE}
    item = out[0]
    assert item.id == "biorxiv:10.1/abc"
    assert item.url == "https://doi.org/10.1/abc"
    assert item.text == "Abs"  # abstract becomes text


# ----- twitter -----


def test_twitter_iterates_every_handle(monkeypatch):
    calls = []

    def _recent_tweets(handle):
        calls.append(handle)
        return [{"id": f"{handle}1", "createdAtISO": "2024-01-01T00:00:00Z", "text": "hi"}]

    monkeypatch.setattr(adapters.x, "recent_tweets", _recent_tweets)

    out = twitter(_spec(handles=["h1", "h2"]), _SINCE, {})

    assert calls == ["h1", "h2"]
    assert [i.id for i in out] == ["x:h11", "x:h21"]
    assert {i.meta["handle"] for i in out} == {"h1", "h2"}


# == Enrichers ================================================================

# ----- article_text -----


def test_article_text_replaces_when_body_present(monkeypatch):
    monkeypatch.setattr(adapters.url, "article_text", lambda _u: "full body")

    out = article_text(_item(text="orig"))

    assert out.text == "full body"


@pytest.mark.parametrize("body", [None, ""], ids=["none", "empty"])
def test_article_text_leaves_original_when_no_body(monkeypatch, body):
    monkeypatch.setattr(adapters.url, "article_text", lambda _u: body)

    out = article_text(_item(text="orig"))

    assert out.text == "orig"


# == Helper Functions =========================================================

# ----- _tweet_item -----


@pytest.mark.parametrize(
    "tweet",
    [
        {"createdAtISO": "2024-01-01T00:00:00Z"},
        {"id": "1"},
        {},
    ],
    ids=["missing_id", "missing_date", "missing_both"],
)
def test_tweet_item_missing_id_or_date_raises(tweet):
    with pytest.raises(x.UnexpectedXFormat):
        _tweet_item(tweet, _spec(), "h")


def test_tweet_item_unparseable_date_chains_value_error():
    tweet = {"id": "1", "createdAtISO": "not-a-date"}

    with pytest.raises(x.UnexpectedXFormat) as ei:
        _tweet_item(tweet, _spec(), "h")

    assert isinstance(ei.value.__cause__, ValueError)


def test_tweet_item_z_suffix_normalized_and_createdat_fallback_accepted():
    tweet = {"id": "42", "createdAt": "2024-06-01T10:00:00Z", "text": "hello"}

    item = _tweet_item(tweet, _spec(), "h")

    assert item.id == "x:42"
    assert item.published == datetime(2024, 6, 1, 10, 0, tzinfo=UTC)


def test_tweet_item_missing_url_falls_back_and_title_truncated():
    long_text = "x" * 100
    tweet = {"id": "7", "createdAtISO": "2024-01-01T00:00:00Z", "text": long_text}

    item = _tweet_item(tweet, _spec(), "handle1")

    assert item.url == "https://x.com/handle1/status/7"
    assert item.title == long_text[:80]
    assert len(item.title) == 80
