"""Podcast queue pop/removal + grounded research + episode generation. _generate_episode is
stubbed wholesale for the queue cases; under test, its collaborators (content_generator.research,
llm.resolve_models, podcastfy.client.generate_podcast) are stubbed individually."""

import logging

import pytest

from src.core import llm
from src.core.models import Context
from src.tasks.podcast import task as podcast_task
from src.tasks.podcast.task import (
    DONE_KEY,
    MAX_SOURCE_CHARS,
    _episode_text,
    _generate_episode,
    _is_tts_failure,
    _research,
    run,
)

_TOPICS = ["PROTACs", "ADCs", "mRNA"]
_OVERVIEW = "grounded overview of the topic"
_KEY = "GOOGLE_AI_STUDIO_KEY"


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


def _state(done=None):
    kv = {} if done is None else {DONE_KEY: list(done)}
    return {"ids": {}, "kv": kv}


def _stub_generate(monkeypatch, result):
    monkeypatch.setattr(podcast_task, "_generate_episode", lambda ctx, topic: result)


# ----- run: queue behavior -----


def test_run_airs_first_pending_topic_and_marks_it_done(monkeypatch):
    _stub_generate(monkeypatch, "/tmp/ep.mp3")
    state = _state()  # nothing aired -> pending is all of TOPICS, in order
    result = run(_ctx(state))
    assert result.meta["topic"] == "PROTACs"
    assert result.artifacts == ["/tmp/ep.mp3"]
    assert state["kv"][DONE_KEY] == ["PROTACs"]  # marked aired


def test_run_skips_already_aired_topics(monkeypatch):
    _stub_generate(monkeypatch, "/tmp/ep.mp3")
    state = _state(["PROTACs", "ADCs"])  # first two aired -> next pending is mRNA
    result = run(_ctx(state))
    assert result.meta["topic"] == "mRNA"
    assert set(state["kv"][DONE_KEY]) == {"PROTACs", "ADCs", "mRNA"}


def test_run_all_topics_aired_is_noop(monkeypatch):
    calls = {"n": 0}

    def _gen(ctx, topic):
        calls["n"] += 1
        return "/tmp/ep.mp3"

    monkeypatch.setattr(podcast_task, "_generate_episode", _gen)
    result = run(_ctx(_state(_TOPICS)))  # every topic already aired
    assert result.markdown == "" and not result.artifacts
    assert calls["n"] == 0


def test_run_generation_failure_does_not_mark_done(monkeypatch):
    _stub_generate(monkeypatch, None)
    state = _state()
    result = run(_ctx(state))
    assert result.artifacts == []
    assert DONE_KEY not in state["kv"]  # not aired -> stays pending, retried next run


def test_run_generation_failure_records_a_notice(monkeypatch):
    """Without a notice the Result records nothing, hiding the failure entirely."""
    _stub_generate(monkeypatch, None)
    result = run(_ctx(_state()))
    assert result.notices == [podcast_task.NO_EPISODE_NOTICE]


# ----- research -----


def _stub_research(monkeypatch, result=None, raises=None):
    def _research_impl(topic, *, api_key):
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(podcast_task.content_generator, "research", _research_impl)


def test_research_returns_the_grounded_overview(monkeypatch):
    monkeypatch.setenv(_KEY, "k")
    _stub_research(monkeypatch, result=_OVERVIEW)
    assert _research(_ctx(_state()), "PROTACs") == _OVERVIEW


def test_research_passes_the_topic_and_ai_studio_key(monkeypatch):
    monkeypatch.setenv(_KEY, "the-key")
    seen = {}

    def _research_impl(topic, *, api_key):
        seen.update(topic=topic, api_key=api_key)
        return _OVERVIEW

    monkeypatch.setattr(podcast_task.content_generator, "research", _research_impl)
    _research(_ctx(_state()), "PROTACs")
    assert seen == {"topic": "PROTACs", "api_key": "the-key"}


@pytest.mark.parametrize(
    "key_set, raises",
    [(False, None), (True, RuntimeError("403 blocked")), (True, None)],
    ids=["no_key", "sdk_raises", "empty_text"],
)
def test_research_degrades_to_empty_string(monkeypatch, key_set, raises):
    """Every failure path yields "", which the caller turns into a skipped episode rather than
    an episode built on nothing."""
    monkeypatch.delenv(_KEY, raising=False)
    if key_set:
        monkeypatch.setenv(_KEY, "k")
    _stub_research(monkeypatch, result="", raises=raises)
    assert _research(_ctx(_state()), "PROTACs") == ""


# ----- episode text -----


def test_episode_text_leads_with_topic_and_caps_the_research_body():
    body = "x" * (MAX_SOURCE_CHARS + 500)
    text = _episode_text("PROTACs", body)
    assert text.startswith("PROTACs")
    assert len(text) < len("PROTACs") + len(body)  # research body truncated


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Failed to generate audio: 400 input.text is longer than the limit", True),
        ("Error converting text to speech: boom", True),
        ("litellm.APIError: OpenrouterException - ResourceExhausted", False),
        ("403 Gemini API has not been used in project", False),
    ],
    ids=["byte_limit", "tts_wrapper", "llm_rate_limit", "llm_auth"],
)
def test_is_tts_failure_distinguishes_audio_from_transcript_failures(message, expected):
    assert _is_tts_failure(RuntimeError(message)) is expected


# ----- _generate_episode -----


def _stub_episode_collaborators(
    monkeypatch, *, models, generate_podcast, overview=_OVERVIEW, raises=None
):
    """Stub content_generator.research, llm.resolve_models, and generate_podcast (patched on the
    podcastfy module, since it is imported locally inside the function)."""
    monkeypatch.setenv(_KEY, "k")  # research needs it, so Gemini also leads the cascade
    _stub_research(monkeypatch, result=overview, raises=raises)
    monkeypatch.setattr(podcast_task.llm, "resolve_models", lambda podcast=None: models)

    import podcastfy.client

    monkeypatch.setattr(podcastfy.client, "generate_podcast", generate_podcast)


def test_generate_episode_returns_none_when_generate_podcast_raises(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("boom")

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/some-model"], generate_podcast=_raise
    )
    assert _generate_episode(_ctx(_state()), "PROTACs") is None


def test_generate_episode_passes_topic_anchored_research_and_no_urls(monkeypatch):
    """podcastfy gets the research text only — a URL list here would have it crawl the web
    itself, outside the grounded search."""
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/some-model"], generate_podcast=_capture
    )
    result = _generate_episode(_ctx(_state()), "PROTACs")
    assert result == "/tmp/ep.mp3"
    assert captured["urls"] is None
    assert captured["text"].startswith("PROTACs")
    assert _OVERVIEW in captured["text"]
    assert captured["tts_model"] == podcast_task._TTS_MODEL == "gemini"


def test_generate_episode_caps_podcastfy_per_part_output_tokens(monkeypatch):
    """podcastfy instructs every longform part to fill max_output_tokens, so its 8192 default
    is the length lever — leaving it alone is what produced a 3h17 episode."""
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/some-model"], generate_podcast=_capture
    )
    _generate_episode(_ctx(_state()), "PROTACs")

    generator = captured["config"].get("content_generator")
    assert generator["max_output_tokens"] == podcast_task._MAX_OUTPUT_TOKENS
    assert generator["longform_prompt_template"]  # untouched keys survive the override
    assert captured["conversation_config"]["max_num_chunks"] == podcast_task._MAX_NUM_CHUNKS


@pytest.mark.parametrize(
    "overview, raises",
    [("", None), (None, RuntimeError("403 blocked"))],
    ids=["no_research_text", "research_raises"],
)
def test_generate_episode_skips_when_research_yields_nothing(monkeypatch, overview, raises):
    calls = []

    def _capture(**kwargs):
        calls.append(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch,
        models=["openrouter/some-model"],
        generate_podcast=_capture,
        overview=overview,
        raises=raises,
    )
    assert _generate_episode(_ctx(_state()), "PROTACs") is None
    assert calls == []  # never reached transcript generation


def test_generate_episode_leads_the_cascade_with_gemini(monkeypatch):
    """Research needs the AI Studio key, so whenever an episode is possible at all the Gemini
    transcript candidate is available too, and it goes first."""
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/free-fallback"], generate_podcast=_capture
    )
    _generate_episode(_ctx(_state()), "PROTACs")
    assert captured["llm_model_name"] == podcast_task._TRANSCRIPT_MODEL
    assert captured["api_key_label"] == _KEY


def test_generate_episode_cascades_to_next_model_when_one_fails(monkeypatch):
    attempts = []

    def _capture(**kwargs):
        attempts.append(kwargs["llm_model_name"])
        if len(attempts) == 1:
            raise RuntimeError("upstream rate limit")
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/first", "openrouter/second"], generate_podcast=_capture
    )
    result = _generate_episode(_ctx(_state()), "PROTACs")
    assert result == "/tmp/ep.mp3"
    assert attempts[0] == podcast_task._TRANSCRIPT_MODEL  # Gemini first, then OpenRouter
    assert attempts[1] == "openrouter/first"


def test_generate_episode_stops_the_cascade_on_an_audio_failure(monkeypatch):
    """Regenerating the transcript cannot fix the audio layer; trying cost 30+ minutes."""
    attempts = []

    def _raise(**kwargs):
        attempts.append(kwargs["llm_model_name"])
        raise RuntimeError("Failed to generate audio: 400 input.text is longer than the limit")

    _stub_episode_collaborators(
        monkeypatch,
        models=["openrouter/first", "openrouter/second", "openrouter/third"],
        generate_podcast=_raise,
    )
    assert _generate_episode(_ctx(_state()), "PROTACs") is None
    assert len(attempts) == 1  # stopped after the audio failure


def test_generate_episode_returns_none_when_all_models_fail(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("upstream rate limit")

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/a", "openrouter/b"], generate_podcast=_raise
    )
    assert _generate_episode(_ctx(_state()), "PROTACs") is None


def test_generate_episode_falls_back_to_fallback_model_when_resolve_models_empty(monkeypatch):
    attempts = []

    def _capture(**kwargs):
        attempts.append(kwargs["llm_model_name"])
        raise RuntimeError("upstream rate limit")

    _stub_episode_collaborators(monkeypatch, models=[], generate_podcast=_capture)
    _generate_episode(_ctx(_state()), "PROTACs")
    assert llm.FALLBACK_MODEL in attempts
