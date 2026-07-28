"""Podcast task: generate a two-host episode for the next unaired topic via podcastfy
(Gemini 3.1 Flash transcript with an OpenRouter fallback; Google Cloud TTS for audio)."""

import asyncio
import os
import re
from pathlib import Path

from src.core import llm, sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks
from src.fetchers.url import article_text
from src.tasks.podcast.utils import reachable_urls

DONE_KEY = "podcast_done"
TOPICS: list[str] = sources_loader.load(Path(__file__).parent, []) or []
MAX_SOURCE_URLS = 10
MAX_SOURCE_CHARS = 12000  # bounds the transcript input, which drives podcastfy's part count
_MAX_MODEL_ATTEMPTS = 4
_SOURCE_SEPARATOR = "\n\n"
_TRANSCRIPT_MODEL = "gemini/gemini-3.1-flash"
_GOOGLE_AI_STUDIO_KEY_LABEL = "GOOGLE_AI_STUDIO_KEY"
_OPENROUTER_KEY_LABEL = "OPENROUTER_API_KEY"
_TTS_MODEL = "gemini"
DISCOVER_PROMPT = (Path(__file__).parent / "source_discovery_prompt.md").read_text()
RELEVANCE_PROMPT = (Path(__file__).parent / "relevance_prompt.md").read_text()
INSTRUCTIONS = (Path(__file__).parent / "prompt.md").read_text()
NO_EPISODE_NOTICE = (
    "No episode today — no source could be verified as relevant to the topic, or generation "
    "failed. The topic stays queued and is retried on the next run."
)
_MIN_RELEVANT_SOURCES = 1  # below this the episode is skipped, not built on unverified sources
_DISCOVERY_ATTEMPTS = 2
_JUDGE_EXCERPT_CHARS = 1200
# Episode length: podcastfy instructs each longform part to fill max_output_tokens, so the cap
# is the length lever. ~1200 tokens x ~9 parts ≈ 7k words ≈ 45 min; its default 8192 gave 3h17.
_MAX_OUTPUT_TOKENS = 1200
_MAX_NUM_CHUNKS = 8
_MIN_CHUNK_SIZE = 600
CONVERSATION_CONFIG = {
    "conversation_style": ["technical", "narrative", "engaging", "story-driven"],
    "roles_person1": "curious host who keeps one narrative thread going with sharp questions",
    "roles_person2": "expert who explains via cause-and-effect and vivid examples, not lists",
    "dialogue_structure": [
        "The hook and why it matters",
        "How it actually works",
        "Tradeoffs, failure modes, and edge cases",
        "Where it's heading",
    ],
    "podcast_name": "Daily Podcast",
    "podcast_tagline": "A daily podcast",
    "output_language": "English",
    "engagement_techniques": ["analogies", "worked examples", "storytelling", "callbacks"],
    "creativity": 0.7,
    "max_num_chunks": _MAX_NUM_CHUNKS,
    "min_chunk_size": _MIN_CHUNK_SIZE,
    "text_to_speech": {
        "gemini": {
            "default_voices": {
                "question": "en-US-Chirp3-HD-Iapetus",
                "answer": "en-US-Chirp3-HD-Laomedeia",
            },
        },
    },
}


# == Task =====================================================================


@tasks.register("podcast")
def run(ctx: Context) -> Result:
    """Generate the next unaired topic from sources.yaml; mark it aired on success.

    sources.yaml (TOPICS) is the source of truth for content and order; state only records
    which topics have aired."""
    done = set(state_mod.get_kv(ctx.state, DONE_KEY, []))
    pending = [t for t in TOPICS if t not in done]
    if not pending:
        ctx.logger.info("podcast: all topics aired — nothing to generate")
        return Result(subject="Podcast — (all topics aired)", markdown="")

    topic = pending[0]
    ctx.logger.info(f"podcast: topic={topic!r} ({len(pending)} pending)")
    subject = f"Podcast — {topic}"
    audio_path = asyncio.run(_generate_episode(ctx, topic))
    if audio_path is None:
        # a notice, not an empty Result: an empty one records nothing at all
        return Result(
            subject=subject, markdown="", meta={"topic": topic}, notices=[NO_EPISODE_NOTICE]
        )

    state_mod.set_kv(ctx.state, DONE_KEY, sorted(done | {topic}))  # mark aired only on success
    return Result(subject=subject, markdown="", artifacts=[audio_path], meta={"topic": topic})


# == Source discovery =========================================================


def _discover_urls(ctx: Context, topic: str, *, exclude: list[str] | None = None) -> list[str]:
    """Ask the LLM for candidate source URLs, capped at MAX_SOURCE_URLS.

    Excluded URLs are listed back to the model so a retry proposes new candidates, and
    filtered from the reply too, since the model may ignore the instruction."""
    user = topic
    if exclude:
        user = f"{topic}\n\nDo not return any of these URLs:\n" + "\n".join(exclude)
    raw = ctx.call(system=DISCOVER_PROMPT, user=user)
    urls = [line.strip() for line in raw.splitlines() if line.strip().startswith("http")]
    excluded = set(exclude or ())
    return [url for url in urls if url not in excluded][:MAX_SOURCE_URLS]


async def _relevant_sources(ctx: Context, topic: str) -> list[tuple[str, str]]:
    """(url, text) pairs the judge accepted as on-topic; [] when nothing survives.

    Discovery is retried excluding everything already tried, since a hallucinated URL set
    varies between draws."""
    tried: list[str] = []
    for attempt in range(1, _DISCOVERY_ATTEMPTS + 1):
        urls = await reachable_urls(_discover_urls(ctx, topic, exclude=tried))
        ctx.logger.info(
            "podcast: discovery %d/%d — %d reachable url(s): %s",
            attempt,
            _DISCOVERY_ATTEMPTS,
            len(urls),
            ", ".join(urls) or "none",
        )
        if not urls:
            continue
        tried.extend(urls)
        extracted = await _extract_sources(urls)
        if not extracted:
            ctx.logger.warning(
                "⚠️ podcast: no text extracted from %d reachable url(s); nothing to judge",
                len(urls),
            )
            continue
        kept = _judge_sources(ctx, topic, extracted)
        if kept:
            return kept
    return []


def _judge_sources(
    ctx: Context, topic: str, sources: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """The sources the LLM judged on-topic, decided in one call over all candidates.

    Judges the extracted text, not the URL: a hallucinated identifier resolves to a real
    page, so only the body reveals the mismatch."""
    listing = _SOURCE_SEPARATOR.join(
        f"[{i}] {url}\n{text[:_JUDGE_EXCERPT_CHARS]}" for i, (url, text) in enumerate(sources, 1)
    )
    try:
        reply = ctx.call(system=RELEVANCE_PROMPT, user=f"TOPIC: {topic}\n\nSOURCES:\n{listing}")
    except Exception as exc:  # degrade to no sources, never to unjudged ones
        ctx.logger.warning("⚠️ podcast: judge unavailable (%s); dropping all sources", exc)
        return []

    keep = _parse_keep_indices(reply, len(sources))
    for i, (url, _) in enumerate(sources, 1):
        ctx.logger.info("podcast: source %s %s", "kept" if i in keep else "dropped", url)
    return [source for i, source in enumerate(sources, 1) if i in keep]


def _parse_keep_indices(reply: str, count: int) -> set[int]:
    """1-based indices named in the judge's reply, ignoring anything out of range.

    An empty set is a valid verdict ("NONE", or nothing parsable): drop every source."""
    return {i for i in (int(n) for n in re.findall(r"\d+", reply)) if 1 <= i <= count}


# == Episode generation =======================================================


async def _generate_episode(ctx: Context, topic: str) -> str | None:
    """Episode from topic-anchored, judged sources; None if none survive or every model fails."""
    sources = await _relevant_sources(ctx, topic)
    if len(sources) < _MIN_RELEVANT_SOURCES:
        ctx.logger.warning(
            "⚠️ podcast: %d relevant source(s) for %r, need %d — skipping, topic stays queued",
            len(sources),
            topic,
            _MIN_RELEVANT_SOURCES,
        )
        return None
    ctx.logger.info("podcast: %d relevant source(s) kept for %r", len(sources), topic)

    # imported here, not at module scope, so the task stays importable without podcastfy
    from podcastfy.client import generate_podcast

    config = _podcastfy_config()
    last_err: Exception | None = None
    for model, key_label in _transcript_candidates():
        try:
            return generate_podcast(
                urls=None,  # judged text only: URLs here would be re-crawled unjudged
                text=_episode_text(topic, sources),
                config=config,
                conversation_config={**CONVERSATION_CONFIG, "user_instructions": INSTRUCTIONS},
                llm_model_name=model,
                api_key_label=key_label,
                tts_model=_TTS_MODEL,
                longform=True,
            )
        except Exception as exc:
            last_err = exc
            ctx.logger.warning("⚠️ podcast: model=%s failed: %s", model, exc)
    ctx.logger.warning("⚠️ podcast: no episode produced for %r: %s", topic, last_err)
    return None


def _transcript_candidates() -> list[tuple[str, str]]:
    """(model, key-env) pairs to try: Gemini when its key is set, then the OpenRouter fallback."""
    candidates: list[tuple[str, str]] = []
    if os.environ.get(_GOOGLE_AI_STUDIO_KEY_LABEL):
        candidates.append((_TRANSCRIPT_MODEL, _GOOGLE_AI_STUDIO_KEY_LABEL))
    free = llm.resolve_models(podcast=True) or [llm.FALLBACK_MODEL]
    candidates += [(m, _OPENROUTER_KEY_LABEL) for m in free]
    return candidates[:_MAX_MODEL_ATTEMPTS]


async def _extract_sources(urls: list[str]) -> list[tuple[str, str]]:
    """(url, article body) for each URL that yielded text; URLs yielding none are dropped."""
    texts = await asyncio.gather(*(asyncio.to_thread(article_text, url) for url in urls))
    return [(url, text) for url, text in zip(urls, texts, strict=True) if text]


def _episode_text(topic: str, sources: list[tuple[str, str]]) -> str:
    """The transcript input: topic first so the episode stays anchored to it, then the
    judged source bodies. Source text alone let the episode drift to whatever it described."""
    body = _SOURCE_SEPARATOR.join(text for _, text in sources)
    return f"{topic}{_SOURCE_SEPARATOR}{body[:MAX_SOURCE_CHARS]}"


def _podcastfy_config():
    """podcastfy's Config with the per-part output cap lowered from its 8192 default."""
    from podcastfy.utils.config import load_config

    config = load_config()
    generator = {**config.get("content_generator", {}), "max_output_tokens": _MAX_OUTPUT_TOKENS}
    config.configure(content_generator=generator)
    return config
