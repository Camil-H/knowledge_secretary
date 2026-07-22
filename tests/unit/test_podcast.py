"""Podcast queue pop/removal + URL discovery/validation. podcastfy and the real
network are never touched — _generate_episode is stubbed for the queue cases,
ctx.call is faked for discovery, and httpx.AsyncClient is faked for validation."""

import asyncio

import pytest

from src.core.models import Context
from src.tasks.podcast import task as podcast_task
from src.tasks.podcast.task import MAX_SOURCE_URLS, QUEUE_KEY, _discover_urls, _validate_urls, run

_TOPICS = ["PROTACs", "ADCs", "mRNA"]


@pytest.fixture(autouse=True)
def _patch_topics(monkeypatch):
    monkeypatch.setattr(podcast_task, "TOPICS", _TOPICS)


def _ctx(state, call=None):
    return Context(
        state=state,
        gather=lambda specs, since: [],
        call=call or (lambda tier, system, user, max_tokens=None: ""),
        log=lambda m: None,
    )


def _state(queue=None):
    kv = {} if queue is None else {QUEUE_KEY: list(queue)}
    return {"ids": {}, "kv": kv}


def _stub_generate(monkeypatch, result):
    async def _gen(topic, ctx):
        return result

    monkeypatch.setattr(podcast_task, "_generate_episode", _gen)


# ----- run: queue behavior -----


def test_run_pops_first_topic_and_removes_it_on_success(monkeypatch):
    _stub_generate(monkeypatch, "/tmp/ep.mp3")
    state = _state()  # unseeded -> queue seeded from TOPICS
    result = run(_ctx(state))
    assert result.meta["topic"] == "PROTACs"
    assert result.artifacts == ["/tmp/ep.mp3"]
    assert state["kv"][QUEUE_KEY] == ["ADCs", "mRNA"]  # generated topic removed


def test_run_uses_the_persisted_queue(monkeypatch):
    _stub_generate(monkeypatch, "/tmp/ep.mp3")
    state = _state(["mRNA"])
    result = run(_ctx(state))
    assert result.meta["topic"] == "mRNA"
    assert state["kv"][QUEUE_KEY] == []


def test_run_empty_queue_is_noop(monkeypatch):
    calls = {"n": 0}

    async def _gen(topic, ctx):
        calls["n"] += 1
        return "/tmp/ep.mp3"

    monkeypatch.setattr(podcast_task, "_generate_episode", _gen)
    result = run(_ctx(_state([])))
    assert result.markdown == "" and not result.artifacts
    assert calls["n"] == 0


def test_run_generation_failure_keeps_topic(monkeypatch):
    _stub_generate(monkeypatch, None)
    state = _state()
    result = run(_ctx(state))
    assert result.artifacts == []
    assert QUEUE_KEY not in state["kv"]  # queue not advanced -> topic retried next run


# ----- discovery + validation -----


def test_discover_urls_extracts_links_and_caps():
    reply = "https://a.com\nnot a url\nhttps://b.org\n  https://c.net  \n" + "\n".join(
        f"https://x{i}.com" for i in range(10)
    )
    ctx = _ctx(_state(), call=lambda tier, system, user, max_tokens=None: reply)
    urls = _discover_urls(ctx, "PROTACs")
    assert urls[:3] == ["https://a.com", "https://b.org", "https://c.net"]
    assert len(urls) == MAX_SOURCE_URLS
    assert "not a url" not in urls


def test_validate_urls_drops_unreachable(monkeypatch):
    class _Resp:
        def __init__(self, code):
            self.status_code = code

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def head(self, url):
            return _Resp(200 if "ok" in url else 404)

        async def get(self, url):
            return _Resp(404)

    monkeypatch.setattr(podcast_task.httpx, "AsyncClient", lambda *a, **k: _FakeClient())
    urls = asyncio.run(_validate_urls(["https://ok.com", "https://bad.com"]))
    assert urls == ["https://ok.com"]
