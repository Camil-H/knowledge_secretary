# tests/unit/fetchers/test_url.py
"""Deterministic article-text logic. The redirect-following fetch and trafilatura
are stubbed — this module tests only what article_text does with their results."""

import logging

import httpx
import pytest

from src.core.url_guard import UnsafeURLError
from src.fetchers import url as url_fetcher
from tests.unit.fetchers.conftest import _FakeHttpResp, _raiser

# == url.article_text =========================================================


def _patch_fetch(monkeypatch, result):
    fetch = result if callable(result) else lambda _u: result
    monkeypatch.setattr(url_fetcher, "fetch_following_safe_redirects", fetch)


def test_article_text_returns_extracted_text_on_success(monkeypatch):
    _patch_fetch(monkeypatch, _FakeHttpResp(text="<html>raw</html>"))
    monkeypatch.setattr(url_fetcher.trafilatura, "extract", lambda d: f"extracted:{d}")

    assert url_fetcher.article_text("http://x") == "extracted:<html>raw</html>"


@pytest.mark.parametrize(
    "result",
    [
        pytest.param(_FakeHttpResp(text=""), id="empty_body"),
        pytest.param(_FakeHttpResp(text="<html>err</html>", status_code=500), id="error_status"),
        pytest.param(_raiser(UnsafeURLError("non-public host")), id="unsafe_hop"),
        pytest.param(_raiser(httpx.TimeoutException("timed out")), id="transport_error"),
    ],
)
def test_article_text_degrades_to_none_without_extracting(monkeypatch, caplog, result):
    _patch_fetch(monkeypatch, result)
    monkeypatch.setattr(
        url_fetcher.trafilatura, "extract", _raiser(AssertionError("must not extract"))
    )

    with caplog.at_level(logging.WARNING):
        assert url_fetcher.article_text("http://x") is None
    assert sum("degraded" in r.message for r in caplog.records) == 1


def test_article_text_none_when_extraction_itself_raises(monkeypatch):
    _patch_fetch(monkeypatch, _FakeHttpResp(text="<html>raw</html>"))
    monkeypatch.setattr(url_fetcher.trafilatura, "extract", _raiser(RuntimeError("boom")))
    assert url_fetcher.article_text("http://x") is None
