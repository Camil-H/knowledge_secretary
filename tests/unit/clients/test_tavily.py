"""The Tavily search transport. httpx.post is faked — no real request or key."""

import httpx
import pytest

import src.clients.tavily as tavily
from src import config
from src.core.errors import AuthError, ExternalError

# ----- test doubles -----

_QUERY = "PROTAC degraders"
_ATTEMPTS = max(config.TAVILY_RETRIES, 1)


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
    """Replace httpx.post with a recorder replaying `response` — or, for a list, each item in
    turn with the last repeating; returns the list its kwargs land in."""
    script = response if isinstance(response, list) else [response]
    calls: list[dict] = []

    def _post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        item = script[min(len(calls) - 1, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(tavily.httpx, "post", _post)
    return calls


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """A key to send, and no backoff sleep so retries don't slow the suite."""
    monkeypatch.setenv(config.TAVILY_KEY_LABEL, "tvly-test")
    monkeypatch.setattr(tavily.time, "sleep", lambda _s: None)


# ===== Primitive =====


def test_search_composes_the_request(monkeypatch):
    """The request is the function's own responsibility: the depth, the result cap and
    include_raw_content decide credit cost and whether page text comes back at all, and the
    explicit per-phase timeout is what keeps retries x timeout inside the job's budget."""
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
    assert calls[0]["timeout"].read == config.TAVILY_TIMEOUT_S
    assert calls[0]["timeout"].connect == config.TAVILY_CONNECT_TIMEOUT_S


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
    """Whole extracted pages come back; without the cap the first long one fills the prompt and
    every later source is truncated away downstream."""
    body = "x" * (config.TAVILY_MAX_PAGE_CHARS * 3)
    _fake_post(monkeypatch, _Resp(payload={"results": [_result(raw_content=body)]}))

    assert len(tavily.search(_QUERY)[0]["text"]) == config.TAVILY_MAX_PAGE_CHARS


@pytest.mark.parametrize(
    "response, expected, requests",
    [
        pytest.param(_Resp(status_code=401, text="bad key"), AuthError, 1, id="unauthorized"),
        pytest.param(
            _Resp(status_code=429, text="out of credits"), ExternalError, 1, id="no_credits"
        ),
        pytest.param(_Resp(text="<html>"), ExternalError, 1, id="body_not_json"),
        pytest.param(
            _Resp(status_code=500, text="boom"), ExternalError, _ATTEMPTS, id="server_error"
        ),
        pytest.param(
            httpx.ConnectError("no route"), ExternalError, _ATTEMPTS, id="transport_failure"
        ),
        pytest.param(httpx.ReadTimeout("timed out"), ExternalError, _ATTEMPTS, id="timeout"),
    ],
)
def test_search_types_its_failures_and_retries_only_transient_ones(
    monkeypatch, response, expected, requests
):
    """A spent credit balance arrives as a 429 and must stay tolerable, so the caller can degrade;
    it also costs exactly one request, since credits will not come back today. Only a rejected
    credential is an AuthError."""
    calls = _fake_post(monkeypatch, response)

    with pytest.raises(expected):
        tavily.search(_QUERY)

    assert len(calls) == requests


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(httpx.ReadTimeout("timed out"), id="timeout"),
        pytest.param(httpx.ConnectError("no route"), id="transport_failure"),
        pytest.param(_Resp(status_code=503, text="busy"), id="server_error"),
    ],
)
def test_search_retries_a_transient_failure_and_returns_the_pages(monkeypatch, failure):
    """One blip used to cost the day's episode."""
    calls = _fake_post(monkeypatch, [failure, _Resp(payload={"results": [_result()]})])

    assert [p["url"] for p in tavily.search(_QUERY)] == ["https://a.example"]
    assert len(calls) == 2


def test_search_without_a_key_raises_before_requesting(monkeypatch):
    monkeypatch.delenv(config.TAVILY_KEY_LABEL, raising=False)
    calls = _fake_post(monkeypatch, _Resp(payload={"results": [_result()]}))

    with pytest.raises(AuthError, match=config.TAVILY_KEY_LABEL):
        tavily.search(_QUERY)

    assert calls == []
