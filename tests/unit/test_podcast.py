"""Podcast queue pop/removal + URL discovery + episode generation. podcastfy and
the LLM are stubbed via monkeypatch — _generate_episode is stubbed wholesale for
the queue cases, and its own collaborators (validate_urls, llm.resolve_models,
podcastfy.client.generate_podcast) are stubbed individually when it is under
test; ctx.call is faked for discovery."""

import asyncio
import logging

import pytest

from src.core import llm
from src.core.models import Context
from src.tasks.podcast import task as podcast_task
from src.tasks.podcast.task import (
    MAX_SOURCE_URLS,
    QUEUE_KEY,
    _discover_urls,
    _generate_episode,
    run,
)

_TOPICS = ["PROTACs", "ADCs", "mRNA"]


@pytest.fixture(autouse=True)
def _patch_topics(monkeypatch):
    monkeypatch.setattr(podcast_task, "TOPICS", _TOPICS)


def _ctx(state, call=None):
    return Context(
        state=state,
        gather=lambda specs, since: [],
        call=call or (lambda system, user, max_tokens=None: ""),
        logger=logging.getLogger("test"),
    )


def _state(queue=None):
    kv = {} if queue is None else {QUEUE_KEY: list(queue)}
    return {"ids": {}, "kv": kv}


def _stub_generate(monkeypatch, result):
    async def _gen(ctx, topic):
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

    async def _gen(ctx, topic):
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


# ----- discovery -----


def test_discover_urls_extracts_links_and_caps():
    reply = "https://a.com\nnot a url\nhttps://b.org\n  https://c.net  \n" + "\n".join(
        f"https://x{i}.com" for i in range(10)
    )
    ctx = _ctx(_state(), call=lambda system, user, max_tokens=None: reply)
    urls = _discover_urls(ctx, "PROTACs")
    assert urls[:3] == ["https://a.com", "https://b.org", "https://c.net"]
    assert len(urls) == MAX_SOURCE_URLS
    assert "not a url" not in urls


# ----- _generate_episode -----


def _discovery_ctx():
    """ctx whose ctx.call yields a single discoverable URL; _discover_urls itself
    runs for real here — only its downstream collaborators are stubbed."""
    return _ctx(_state(), call=lambda system, user, max_tokens=None: "https://source.example.com")


def _stub_episode_collaborators(monkeypatch, *, validated_urls, models, generate_podcast):
    """Stub _generate_episode's three collaborators: validate_urls (reachability),
    llm.resolve_models (model choice), and podcastfy.client.generate_podcast
    (via sys.modules, since it's imported locally inside the function)."""

    async def _validate(urls):
        return validated_urls

    monkeypatch.setattr(podcast_task, "validate_urls", _validate)
    monkeypatch.setattr(podcast_task.llm, "resolve_models", lambda podcast=None: models)

    import podcastfy.client

    monkeypatch.setattr(podcastfy.client, "generate_podcast", generate_podcast)


def test_generate_episode_returns_none_when_generate_podcast_raises(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("boom")

    _stub_episode_collaborators(
        monkeypatch,
        validated_urls=["https://source.example.com"],
        models=["openrouter/some-model"],
        generate_podcast=_raise,
    )
    result = asyncio.run(_generate_episode(_discovery_ctx(), "PROTACs"))
    assert result is None


@pytest.mark.parametrize(
    "validated_urls, expected_urls, expected_text",
    [
        ([], None, "PROTACs"),
        (["https://source.example.com"], ["https://source.example.com"], None),
    ],
    ids=["no_reachable_urls", "reachable_urls"],
)
def test_generate_episode_passes_urls_or_topic_text_based_on_reachability(
    monkeypatch, validated_urls, expected_urls, expected_text
):
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch,
        validated_urls=validated_urls,
        models=["openrouter/some-model"],
        generate_podcast=_capture,
    )
    result = asyncio.run(_generate_episode(_discovery_ctx(), "PROTACs"))
    assert result == "/tmp/ep.mp3"
    assert captured["urls"] == expected_urls
    assert captured["text"] == expected_text


def test_generate_episode_falls_back_to_fallback_model_when_resolve_models_empty(monkeypatch):
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch,
        validated_urls=["https://source.example.com"],
        models=[],
        generate_podcast=_capture,
    )
    asyncio.run(_generate_episode(_discovery_ctx(), "PROTACs"))
    assert captured["llm_model_name"] == llm.FALLBACK_MODEL
