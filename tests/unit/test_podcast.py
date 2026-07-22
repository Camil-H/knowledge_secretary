"""Podcast topic-queue pop + removal. _generate_episode (podcastfy) is stubbed
for every case, so these tests cover only run()'s queue logic — never real
generation."""

import pytest

from src.core.models import Context
from src.tasks.podcast import task as podcast_task
from src.tasks.podcast.task import QUEUE_KEY, run

_TOPICS = ["PROTACs", "ADCs", "mRNA"]


@pytest.fixture(autouse=True)
def _patch_topics(monkeypatch):
    monkeypatch.setattr(podcast_task, "TOPICS", _TOPICS)


def _ctx(state):
    return Context(
        state=state,
        gather=lambda specs, since: [],
        call=lambda tier, system, user, max_tokens=None: "",
        log=lambda m: None,
    )


def _state(queue=None):
    kv = {} if queue is None else {QUEUE_KEY: list(queue)}
    return {"ids": {}, "kv": kv}


def test_run_pops_first_topic_and_removes_it_on_success(monkeypatch):
    monkeypatch.setattr(podcast_task, "_generate_episode", lambda topic, ctx: "/tmp/ep.mp3")
    state = _state()  # unseeded -> queue seeded from TOPICS
    result = run(_ctx(state))
    assert result.meta["topic"] == "PROTACs"
    assert result.artifacts == ["/tmp/ep.mp3"]
    assert state["kv"][QUEUE_KEY] == ["ADCs", "mRNA"]  # generated topic removed


def test_run_uses_the_persisted_queue(monkeypatch):
    monkeypatch.setattr(podcast_task, "_generate_episode", lambda topic, ctx: "/tmp/ep.mp3")
    state = _state(["mRNA"])
    result = run(_ctx(state))
    assert result.meta["topic"] == "mRNA"
    assert state["kv"][QUEUE_KEY] == []


def test_run_empty_queue_is_noop(monkeypatch):
    calls = {"n": 0}

    def _boom(topic, ctx):  # must never be reached
        calls["n"] += 1
        return "/tmp/ep.mp3"

    monkeypatch.setattr(podcast_task, "_generate_episode", _boom)
    result = run(_ctx(_state([])))
    assert result.markdown == "" and not result.artifacts
    assert calls["n"] == 0


def test_run_generation_failure_keeps_topic(monkeypatch):
    monkeypatch.setattr(podcast_task, "_generate_episode", lambda topic, ctx: None)
    state = _state()
    result = run(_ctx(state))
    assert result.artifacts == []
    assert QUEUE_KEY not in state["kv"]  # queue not advanced -> topic retried next run
