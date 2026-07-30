# tests/unit/fetchers/test_url_guard_in_fetchers.py
"""The SSRF guard contract shared by every fetcher that builds its own URL: the
guard runs before the transport, and a rejection degrades instead of fetching.

This lives outside the per-module test files on purpose — the contract belongs to
no single fetcher, and the matrix is the assertion: a new URL-building fetcher is
covered by adding a row, not by copying the test.
"""

import logging
from datetime import UTC, datetime

import pytest

from src.core.url_guard import UnsafeURLError
from src.fetchers import openrxiv, pubmed, rss
from tests.unit.fetchers.conftest import _raiser

_SINCE = datetime(2024, 1, 1, tzinfo=UTC)

# == The url guard, in every fetcher that applies it ===========================


def _reject_urls(monkeypatch, module) -> list[str]:
    """Make `module`'s guard reject every URL, recording any fetch that slips past it."""
    fetched: list[str] = []
    monkeypatch.setattr(module, "assert_safe_url", _raiser(UnsafeURLError("non-public host")))
    monkeypatch.setattr(module.httpx, "get", lambda url, **_k: fetched.append(url))
    return fetched


@pytest.mark.parametrize(
    ("module", "call", "degraded"),
    [
        pytest.param(
            rss,
            lambda: rss.fetch("http://169.254.169.254/feed"),
            {"title": "", "entries": []},
            id="rss",
        ),
        pytest.param(pubmed, lambda: pubmed.search_recent(["q"], _SINCE), [], id="pubmed"),
        pytest.param(
            openrxiv,
            lambda: openrxiv.recent("biorxiv", ["neuroscience"], _SINCE),
            [],
            id="openrxiv",
        ),
    ],
)
def test_fetcher_degrades_without_fetching_when_the_guard_rejects(
    monkeypatch, caplog, module, call, degraded
):
    fetched = _reject_urls(monkeypatch, module)

    with caplog.at_level(logging.WARNING):
        assert call() == degraded
    assert fetched == []
    assert any("degraded" in r.message for r in caplog.records)
