"""The Tavily search transport. httpx.post is faked — no real request or key."""

import httpx
import pytest

import src.clients.tavily as tavily
from src import config
from src.core.errors import AuthError, ExternalError

# ----- test doubles -----

_QUERY = "PROTAC degraders"


class _Resp:
    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _result(url: str = "https://a.example", **over) -> dict:
    return {"title": "A title", "url": url, "raw_content": "the page body", **over}


def _fake_post(monkeypatch, response) -> list[dict]:
    """Replace httpx.post with a recorder; returns the list its kwargs land in."""
    calls: list[dict] = []

    def _post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(tavily.httpx, "post", _post)
    return calls


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv(config.TAVILY_KEY_LABEL, "tvly-test")


# ===== Primitive =====


def test_search_composes_the_request(monkeypatch):
    """The request is the function's own responsibility: the depth, the result cap and
    include_raw_content are what decide credit cost and whether page text comes back at all."""
    calls = _fake_post(monkeypatch, _Resp(payload={"results": [_result()]}))

    tavily.search(_QUERY)

    assert calls[0]["url"] == tavily.SEARCH_URL
    assert calls[0]["headers"]["Authorization"] == "Bearer tvly-test"
    assert calls[0]["json"] == {
        "query": _QUERY,
        "search_depth": config.TAVILY_SEARCH_DEPTH,
        "max_results": config.TAVILY_MAX_RESULTS,
        "include_raw_content": True,
    }
    assert calls[0]["timeout"] == config.HTTP_TIMEOUT_S


@pytest.mark.parametrize(
    "payload, expected",
    [
        pytest.param(
            {"results": [_result(), _result("https://b.example")]},
            ["https://a.example", "https://b.example"],
            id="keeps_result_order",
        ),
        pytest.param(
            {"results": [_result(raw_content=None, content="the snippet")]},
            ["https://a.example"],
            id="falls_back_to_content_snippet",
        ),
        pytest.param(
            {"results": [_result(raw_content="", content=""), _result("https://b.example")]},
            ["https://b.example"],
            id="drops_a_textless_result",
        ),
        pytest.param(
            {"results": [_result(raw_content="   ")]}, [], id="drops_whitespace_only_text"
        ),
        pytest.param({"results": []}, [], id="no_results"),
        pytest.param({}, [], id="no_results_key"),
        pytest.param({"results": "nope"}, [], id="results_not_a_list"),
        pytest.param({"results": ["nope", _result()]}, ["https://a.example"], id="skips_non_dict"),
        pytest.param([1, 2], [], id="body_not_an_object"),
    ],
)
def test_search_returns_the_usable_pages(monkeypatch, payload, expected):
    _fake_post(monkeypatch, _Resp(payload=payload))

    assert [p["url"] for p in tavily.search(_QUERY)] == expected


def test_search_flattens_a_result_to_title_url_and_text(monkeypatch):
    _fake_post(monkeypatch, _Resp(payload={"results": [_result()]}))

    assert tavily.search(_QUERY) == [
        {"title": "A title", "url": "https://a.example", "text": "the page body"}
    ]


def test_search_caps_one_page_so_it_cannot_crowd_out_the_others(monkeypatch):
    """Whole extracted pages come back; without the cap the first long one fills the research
    prompt and every later source is truncated away downstream."""
    body = "x" * (config.TAVILY_MAX_PAGE_CHARS * 3)
    _fake_post(monkeypatch, _Resp(payload={"results": [_result(raw_content=body)]}))

    assert len(tavily.search(_QUERY)[0]["text"]) == config.TAVILY_MAX_PAGE_CHARS


@pytest.mark.parametrize(
    "response, expected",
    [
        pytest.param(_Resp(status_code=401, text="bad key"), AuthError, id="unauthorized"),
        pytest.param(_Resp(status_code=429, text="out of credits"), ExternalError, id="no_credits"),
        pytest.param(_Resp(status_code=500, text="boom"), ExternalError, id="server_error"),
        pytest.param(httpx.ConnectError("no route"), ExternalError, id="transport_failure"),
        pytest.param(_Resp(text="<html>"), ExternalError, id="body_not_json"),
    ],
)
def test_search_types_its_failures(monkeypatch, response, expected):
    """A spent credit balance arrives as a 429 and must stay tolerable — the caller degrades the
    episode. Only a rejected credential is an AuthError."""
    _fake_post(monkeypatch, response)

    with pytest.raises(expected):
        tavily.search(_QUERY)


def test_search_without_a_key_raises_before_requesting(monkeypatch):
    monkeypatch.delenv(config.TAVILY_KEY_LABEL, raising=False)
    calls = _fake_post(monkeypatch, _Resp(payload={"results": [_result()]}))

    with pytest.raises(AuthError, match=config.TAVILY_KEY_LABEL):
        tavily.search(_QUERY)

    assert calls == []
