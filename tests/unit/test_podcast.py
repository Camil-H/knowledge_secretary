"""Podcast queue pop/removal + grounded research + episode generation. _generate_episode is
stubbed wholesale for the queue cases; under test, its collaborators (content_generator.research,
transcript.generate, podcastfy.client.generate_podcast) are stubbed individually."""

import logging

import pytest

from src.core.models import Context
from src.tasks.podcast import task as podcast_task
from src.tasks.podcast.task import DONE_KEY, _generate_episode, _research, run
from src.tasks.podcast.transcript import TranscriptError

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


# ----- _generate_episode -----

_TRANSCRIPT = "<Person1>Hello there.</Person1>\n<Person2>Glad to be here.</Person2>"


def _stub_episode_collaborators(
    monkeypatch, *, generate_podcast, overview=_OVERVIEW, raises=None, transcript_raises=None
):
    """Stub content_generator.research, transcript.generate, and generate_podcast (patched on the
    podcastfy module, since it is imported locally inside the function)."""
    monkeypatch.setenv(_KEY, "k")  # research needs it
    _stub_research(monkeypatch, result=overview, raises=raises)

    def _generate(topic, research, *, call):
        if transcript_raises is not None:
            raise transcript_raises
        return _TRANSCRIPT

    monkeypatch.setattr(podcast_task.transcript, "generate", _generate)

    import podcastfy.client

    monkeypatch.setattr(podcastfy.client, "generate_podcast", generate_podcast)


def _capturing_podcast(captured, result="/tmp/ep.mp3"):
    def _capture(**kwargs):
        captured.update(kwargs)
        return result

    return _capture


def test_generate_episode_returns_the_synthesized_audio_path(monkeypatch):
    _stub_episode_collaborators(monkeypatch, generate_podcast=_capturing_podcast({}))
    assert _generate_episode(_ctx(_state()), "PROTACs") == "/tmp/ep.mp3"


def test_generate_episode_hands_podcastfy_the_transcript_file_only(monkeypatch):
    """podcastfy is the audio layer now: a text=/urls= payload would have it regenerate the
    transcript (and crawl the web) instead of just synthesizing ours."""
    captured = {}
    _stub_episode_collaborators(monkeypatch, generate_podcast=_capturing_podcast(captured))
    _generate_episode(_ctx(_state()), "PROTACs")

    with open(captured["transcript_file"]) as handle:
        assert handle.read() == _TRANSCRIPT
    assert captured["tts_model"] == podcast_task._TTS_MODEL == "gemini"
    speech = captured["conversation_config"]["text_to_speech"]
    assert speech["ending_message"] == ""
    assert speech["gemini"]["default_voices"] == {
        "question": "en-US-Chirp3-HD-Iapetus",
        "answer": "en-US-Chirp3-HD-Laomedeia",
    }
    assert not {"text", "urls", "longform", "llm_model_name"} & set(captured)


def test_generate_episode_passes_the_context_call_to_the_transcript_layer(monkeypatch):
    """The transcript rides ctx.call, so its model cascade lives in one place — src.core.llm."""
    seen = {}

    def _generate(topic, research, *, call):
        seen.update(topic=topic, research=research, call=call)
        return _TRANSCRIPT

    monkeypatch.setenv(_KEY, "k")
    _stub_research(monkeypatch, result=_OVERVIEW)
    monkeypatch.setattr(podcast_task.transcript, "generate", _generate)

    import podcastfy.client

    monkeypatch.setattr(podcastfy.client, "generate_podcast", _capturing_podcast({}))
    call = _ctx(_state()).call
    _generate_episode(_ctx(_state(), call=call), "PROTACs")
    assert seen == {"topic": "PROTACs", "research": _OVERVIEW, "call": call}


@pytest.mark.parametrize(
    "overview, raises",
    [("", None), (None, RuntimeError("403 blocked"))],
    ids=["no_research_text", "research_raises"],
)
def test_generate_episode_skips_transcript_when_research_yields_nothing(
    monkeypatch, overview, raises
):
    calls = []

    def _generate(topic, research, *, call):
        calls.append(topic)
        return _TRANSCRIPT

    monkeypatch.setenv(_KEY, "k")
    _stub_research(monkeypatch, result=overview, raises=raises)
    monkeypatch.setattr(podcast_task.transcript, "generate", _generate)
    assert _generate_episode(_ctx(_state()), "PROTACs") is None
    assert calls == []  # never reached transcript generation


@pytest.mark.parametrize(
    "transcript_raises, podcast_raises",
    [(TranscriptError("no usable turns in part 3"), None), (None, RuntimeError("tts boom"))],
    ids=["transcript_failed", "audio_failed"],
)
def test_generate_episode_degrades_to_none_when_a_stage_fails(
    monkeypatch, transcript_raises, podcast_raises
):
    def _generate_podcast(**kwargs):
        if podcast_raises is not None:
            raise podcast_raises
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch, generate_podcast=_generate_podcast, transcript_raises=transcript_raises
    )
    assert _generate_episode(_ctx(_state()), "PROTACs") is None
