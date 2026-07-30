"""Podcast queue pop/removal + grounded research + episode generation. _generate_episode is
stubbed wholesale for the queue cases; under test, its collaborators (gemini.call,
transcript.generate, audio.synthesize) are stubbed individually."""

import logging
import os

import pytest

from src.core.errors import AudioError, AuthError
from src.core.models import Context
from src.tasks.podcast import task as podcast_task
from src.tasks.podcast.task import DONE_KEY, _generate_episode, _research, run
from src.tasks.podcast.transcript import TranscriptError

_TOPICS = ["PROTACs", "ADCs", "mRNA"]
_OVERVIEW = "grounded overview of the topic"


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


@pytest.mark.parametrize(
    "done, topic, expected_done",
    [
        ([], "PROTACs", ["PROTACs"]),
        (["PROTACs", "ADCs"], "mRNA", ["ADCs", "PROTACs", "mRNA"]),
    ],
    ids=["none_aired", "some_aired"],
)
def test_run_airs_the_first_unaired_topic_and_records_it(monkeypatch, done, topic, expected_done):
    """Marking is additive and sorted: an unsorted merge would make the aired list order depend
    on set iteration, and a non-additive one would re-air topics."""
    _stub_generate(monkeypatch, "/tmp/ep.mp3")
    state = _state(done)
    result = run(_ctx(state))
    assert result.meta["topic"] == topic
    assert result.artifacts == ["/tmp/ep.mp3"]
    assert state["kv"][DONE_KEY] == expected_done


def test_run_all_topics_aired_is_noop(monkeypatch):
    calls = {"n": 0}

    def _gen(ctx, topic):
        calls["n"] += 1
        return "/tmp/ep.mp3"

    monkeypatch.setattr(podcast_task, "_generate_episode", _gen)
    result = run(_ctx(_state(_TOPICS)))
    assert result.markdown == "" and not result.artifacts
    assert calls["n"] == 0


def test_run_generation_failure_does_not_mark_done(monkeypatch):
    _stub_generate(monkeypatch, None)
    state = _state()
    result = run(_ctx(state))
    assert result.artifacts == []
    assert DONE_KEY not in state["kv"]


def test_run_generation_failure_records_a_notice(monkeypatch):
    """Without a notice the Result records nothing, hiding the failure entirely."""
    _stub_generate(monkeypatch, None)
    result = run(_ctx(_state()))
    assert result.notices == [podcast_task.NO_EPISODE_NOTICE]


# ----- research -----

_LEDGER: dict = {"gemini-2.5-flash": {"period": "2026-07-28", "requests": 3}}


def _stub_research(monkeypatch, result=None, raises=None) -> list[dict]:
    """Replace the shared Gemini entry point and the ledger it is handed; returns the list its
    calls are recorded into."""
    calls: list[dict] = []

    def _call(system, user, max_tokens=None, *, ledger, search=False):
        calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "ledger": ledger,
                "search": search,
            }
        )
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(podcast_task.gemini, "call", _call)
    monkeypatch.setattr(podcast_task.ledger_mod, "load", lambda: _LEDGER)
    return calls


@pytest.mark.parametrize(
    "result, raises, expected",
    [
        (_OVERVIEW, None, _OVERVIEW),
        ("", None, ""),
        (None, RuntimeError("403 blocked"), ""),
        (None, AuthError("google-ai-studio"), ""),
    ],
    ids=["overview", "empty_text", "external_failure", "auth_failure"],
)
def test_research_returns_the_overview_or_degrades_to_empty(monkeypatch, result, raises, expected):
    """Every failure path yields "", which the caller turns into a skipped episode rather than
    an episode built on nothing — including an auth failure, which the transport already
    degraded across tiers before giving up."""
    _stub_research(monkeypatch, result=result, raises=raises)
    assert _research(_ctx(_state()), "PROTACs") == expected


def test_research_asks_for_a_grounded_completion_of_the_topic(monkeypatch):
    """`search=True` is the request: grounding-capable models and the search tool, together. The
    day's ledger rides along or the grounded call escapes the request budget."""
    calls = _stub_research(monkeypatch, result=_OVERVIEW)

    _research(_ctx(_state()), "PROTACs")

    assert calls == [
        {
            "system": podcast_task.RESEARCH_PROMPT,
            "user": "PROTACs",
            "max_tokens": None,
            "ledger": _LEDGER,
            "search": True,
        }
    ]


# ----- _generate_episode -----

_TRANSCRIPT = "<Person1>Hello there.</Person1>\n<Person2>Glad to be here.</Person2>"


def _stub_episode_collaborators(
    monkeypatch, *, synthesize, overview=_OVERVIEW, raises=None, transcript_raises=None
) -> dict:
    """Stub gemini.call, transcript.generate and audio.synthesize; returns what generate saw."""
    _stub_research(monkeypatch, result=overview, raises=raises)
    seen: dict = {}

    def _generate(topic, research, *, call):
        seen.update(topic=topic, research=research, call=call)
        if transcript_raises is not None:
            raise transcript_raises
        return _TRANSCRIPT

    monkeypatch.setattr(podcast_task.transcript, "generate", _generate)
    monkeypatch.setattr(podcast_task.audio, "synthesize", synthesize)
    return seen


def _capturing_synthesize(captured):
    """Identity on the transcript: what comes back reflects exactly what was handed off."""

    def _capture(transcript, out_path, *, ledger):
        captured.update(transcript=transcript, out_path=out_path, ledger=ledger)
        return out_path

    return _capture


def test_generate_episode_returns_the_synthesized_audio_path(monkeypatch):
    captured = {}
    _stub_episode_collaborators(monkeypatch, synthesize=_capturing_synthesize(captured))
    assert _generate_episode(_ctx(_state()), "PROTACs") == captured["out_path"]


def test_generate_episode_hands_each_layer_its_inputs(monkeypatch):
    """The transcript rides ctx.call — its model cascade lives in one place, src.clients.llm — and
    flows in memory to the audio layer, which gets its own writable mp3 path per episode."""
    captured = {}
    seen = _stub_episode_collaborators(monkeypatch, synthesize=_capturing_synthesize(captured))
    ctx = _ctx(_state())

    _generate_episode(ctx, "PROTACs")

    assert seen == {"topic": "PROTACs", "research": _OVERVIEW, "call": ctx.call}
    assert captured["transcript"] == _TRANSCRIPT
    assert os.path.basename(captured["out_path"]) == podcast_task.EPISODE_FILENAME
    assert os.path.isdir(os.path.dirname(captured["out_path"]))
    assert captured["ledger"] is _LEDGER


@pytest.mark.parametrize(
    "overview, raises",
    [("", None), (None, RuntimeError("403 blocked"))],
    ids=["no_research_text", "research_raises"],
)
def test_generate_episode_skips_transcript_when_research_yields_nothing(
    monkeypatch, overview, raises
):
    seen = _stub_episode_collaborators(
        monkeypatch, synthesize=_capturing_synthesize({}), overview=overview, raises=raises
    )
    assert _generate_episode(_ctx(_state()), "PROTACs") is None
    assert seen == {}


@pytest.mark.parametrize(
    "transcript_raises, audio_raises",
    [
        (TranscriptError("no usable turns in part 3"), None),
        (None, AudioError("cloud-tts", detail="ffmpeg not on PATH")),
    ],
    ids=["transcript_failed", "audio_failed"],
)
def test_generate_episode_degrades_to_none_when_a_stage_fails(
    monkeypatch, transcript_raises, audio_raises
):
    def _synthesize(transcript, out_path, *, ledger):
        if audio_raises is not None:
            raise audio_raises
        return out_path

    _stub_episode_collaborators(
        monkeypatch, synthesize=_synthesize, transcript_raises=transcript_raises
    )
    assert _generate_episode(_ctx(_state()), "PROTACs") is None
