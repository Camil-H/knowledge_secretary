"""Podcast queue pop/removal + URL discovery + relevance judging + episode generation.
_generate_episode is stubbed wholesale for the queue cases and its collaborators stubbed
individually when it is under test; ctx.call is faked for both discovery and the judge,
keyed on the system prompt it is handed."""

import asyncio
import logging

import pytest

from src.core import llm
from src.core.models import Context
from src.tasks.podcast import task as podcast_task
from src.tasks.podcast.task import (
    DONE_KEY,
    MAX_SOURCE_CHARS,
    MAX_SOURCE_URLS,
    _discover_urls,
    _episode_text,
    _generate_episode,
    _is_tts_failure,
    _parse_keep_indices,
    run,
)

_TOPICS = ["PROTACs", "ADCs", "mRNA"]
_SOURCE_TEXT = "extracted article body"
_URL = "https://source.example.com"
_JUDGE_KEEP_ALL = "1 2 3 4 5 6 7 8 9 10"
_JUDGE_KEEP_NONE = "NONE"


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
    async def _gen(ctx, topic):
        return result

    monkeypatch.setattr(podcast_task, "_generate_episode", _gen)


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

    async def _gen(ctx, topic):
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


def test_discover_urls_lists_and_filters_excluded_urls():
    prompts = []

    def _call(system, user, max_tokens=None):
        prompts.append(user)
        return "https://old.com\nhttps://new.com"

    urls = _discover_urls(_ctx(_state(), call=_call), "PROTACs", exclude=["https://old.com"])
    assert urls == ["https://new.com"]
    assert "https://old.com" in prompts[0]


# ----- relevance judging -----


@pytest.mark.parametrize(
    "reply, count, expected",
    [
        ("1\n3", 3, {1, 3}),
        ("1, 2", 3, {1, 2}),
        ("NONE", 3, set()),
        ("", 3, set()),
        ("keep source 2 only", 3, {2}),
        ("7", 3, set()),  # out of range is ignored, not clamped
        ("Fields Medal 2026", 3, set()),  # stray numbers can't invent a source
    ],
    ids=["lines", "inline", "none", "empty", "prose", "out_of_range", "stray_number"],
)
def test_parse_keep_indices(reply, count, expected):
    assert _parse_keep_indices(reply, count) == expected


# ----- episode text -----


def test_episode_text_leads_with_topic_and_caps_source_body():
    body = "x" * (MAX_SOURCE_CHARS + 500)
    text = _episode_text("PROTACs", [(_URL, body)])
    assert text.startswith("PROTACs")
    assert len(text) < len("PROTACs") + len(body)  # source body truncated


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


def _episode_ctx(*, discovered=_URL, judge=_JUDGE_KEEP_ALL):
    """ctx answering discovery and the judge separately; both helpers run for real."""

    def _call(system, user, max_tokens=None):
        return judge if system == podcast_task.RELEVANCE_PROMPT else discovered

    return _ctx(_state(), call=_call)


def _stub_episode_collaborators(
    monkeypatch, *, models, generate_podcast, validated_urls=None, article_text=None
):
    """Stub reachable_urls (pass-through when validated_urls is None), article_text,
    llm.resolve_models, and generate_podcast (patched on the module: imported locally)."""

    async def _validate(urls):
        return urls if validated_urls is None else validated_urls

    monkeypatch.setattr(podcast_task, "reachable_urls", _validate)
    monkeypatch.setattr(podcast_task, "article_text", article_text or (lambda url: _SOURCE_TEXT))
    monkeypatch.setattr(podcast_task.llm, "resolve_models", lambda podcast=None: models)
    monkeypatch.delenv(
        "GOOGLE_AI_STUDIO_KEY", raising=False
    )  # default: exercise the OpenRouter path

    import podcastfy.client

    monkeypatch.setattr(podcastfy.client, "generate_podcast", generate_podcast)


def test_generate_episode_returns_none_when_generate_podcast_raises(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("boom")

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/some-model"], generate_podcast=_raise
    )
    assert asyncio.run(_generate_episode(_episode_ctx(), "PROTACs")) is None


def test_generate_episode_passes_topic_anchored_judged_text_and_no_urls(monkeypatch):
    """Passing URLs would have podcastfy re-crawl sources the judge never saw."""
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/some-model"], generate_podcast=_capture
    )
    result = asyncio.run(_generate_episode(_episode_ctx(), "PROTACs"))
    assert result == "/tmp/ep.mp3"
    assert captured["urls"] is None
    assert captured["text"].startswith("PROTACs")
    assert _SOURCE_TEXT in captured["text"]
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
    asyncio.run(_generate_episode(_episode_ctx(), "PROTACs"))

    generator = captured["config"].get("content_generator")
    assert generator["max_output_tokens"] == podcast_task._MAX_OUTPUT_TOKENS
    assert generator["longform_prompt_template"]  # untouched keys survive the override
    assert captured["conversation_config"]["max_num_chunks"] == podcast_task._MAX_NUM_CHUNKS


@pytest.mark.parametrize(
    "judge, validated_urls, article_text",
    [
        (_JUDGE_KEEP_NONE, None, None),
        (_JUDGE_KEEP_ALL, [], None),
        (_JUDGE_KEEP_ALL, None, lambda url: None),
    ],
    ids=["judge_rejects_all", "no_reachable_urls", "no_text_extracted"],
)
def test_generate_episode_skips_when_no_source_survives(
    monkeypatch, judge, validated_urls, article_text
):
    """Every route to zero verified sources skips, rather than falling back to ungrounded
    audio or to podcastfy re-crawling unjudged URLs."""
    calls = []

    def _capture(**kwargs):
        calls.append(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch,
        models=["openrouter/some-model"],
        generate_podcast=_capture,
        validated_urls=validated_urls,
        article_text=article_text,
    )
    result = asyncio.run(_generate_episode(_episode_ctx(judge=judge), "PROTACs"))
    assert result is None
    assert calls == []  # never reached transcript generation


def test_generate_episode_retries_discovery_excluding_the_rejected_urls(monkeypatch):
    bad, good = "https://bad.example.com", "https://good.example.com"
    discovery_prompts = []

    def _call(system, user, max_tokens=None):
        if system == podcast_task.RELEVANCE_PROMPT:
            return _JUDGE_KEEP_NONE if bad in user else "1"
        discovery_prompts.append(user)
        return bad if len(discovery_prompts) == 1 else good

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/some-model"], generate_podcast=lambda **kw: "/tmp/ep.mp3"
    )
    result = asyncio.run(_generate_episode(_ctx(_state(), call=_call), "PROTACs"))
    assert result == "/tmp/ep.mp3"
    assert bad in discovery_prompts[1]  # second draw told not to repeat the rejected URL


def test_generate_episode_extracts_sources_once_across_model_retries(monkeypatch):
    extract_calls = []

    def _article_text(url):
        extract_calls.append(url)
        return _SOURCE_TEXT

    attempts = []

    def _capture(**kwargs):
        attempts.append(kwargs["text"])
        if len(attempts) == 1:
            raise RuntimeError("upstream rate limit")
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch,
        models=["openrouter/first", "openrouter/second"],
        generate_podcast=_capture,
        article_text=_article_text,
    )
    result = asyncio.run(_generate_episode(_episode_ctx(), "PROTACs"))
    assert result == "/tmp/ep.mp3"
    assert len(extract_calls) == 1  # fetched once, not once per model attempt
    assert attempts[0] == attempts[1]  # same extracted text reused on retry


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
    result = asyncio.run(_generate_episode(_episode_ctx(), "PROTACs"))
    assert result == "/tmp/ep.mp3"
    assert attempts == ["openrouter/first", "openrouter/second"]


def test_generate_episode_prefers_gemini_when_ai_studio_key_set(monkeypatch):
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(
        monkeypatch, models=["openrouter/free-fallback"], generate_podcast=_capture
    )
    monkeypatch.setenv("GOOGLE_AI_STUDIO_KEY", "ai-studio-key")  # re-add after the stub's delenv

    asyncio.run(_generate_episode(_episode_ctx(), "PROTACs"))

    assert captured["llm_model_name"] == podcast_task._TRANSCRIPT_MODEL
    assert captured["api_key_label"] == "GOOGLE_AI_STUDIO_KEY"


def test_generate_episode_returns_none_when_all_models_fail(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("upstream rate limit")

    _stub_episode_collaborators(
        monkeypatch,
        models=["openrouter/a", "openrouter/b", "openrouter/c"],
        generate_podcast=_raise,
    )
    assert asyncio.run(_generate_episode(_episode_ctx(), "PROTACs")) is None


def test_generate_episode_falls_back_to_fallback_model_when_resolve_models_empty(monkeypatch):
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return "/tmp/ep.mp3"

    _stub_episode_collaborators(monkeypatch, models=[], generate_podcast=_capture)
    asyncio.run(_generate_episode(_episode_ctx(), "PROTACs"))
    assert captured["llm_model_name"] == llm.FALLBACK_MODEL


def test_generate_episode_drops_all_sources_when_the_judge_is_unavailable(monkeypatch):
    """Failing open would restore the very path that let an off-topic source set through."""

    def _call(system, user, max_tokens=None):
        if system == podcast_task.RELEVANCE_PROMPT:
            raise RuntimeError("all models failed")
        return _URL

    calls = []
    _stub_episode_collaborators(
        monkeypatch,
        models=["openrouter/some-model"],
        generate_podcast=lambda **kw: calls.append(kw) or "/tmp/ep.mp3",
    )
    assert asyncio.run(_generate_episode(_ctx(_state(), call=_call), "PROTACs")) is None
    assert calls == []


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
    assert asyncio.run(_generate_episode(_episode_ctx(), "PROTACs")) is None
    assert attempts == ["openrouter/first"]  # stopped after the audio failure
