"""Podcast task: generate a two-host episode for the next unaired topic via podcastfy
(Gemini 3.1 Flash transcript with an OpenRouter fallback; Google Cloud TTS for audio)."""

import asyncio
import os
from pathlib import Path

from src.core import llm, sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks
from src.fetchers.url import article_text
from src.tasks.podcast.utils import reachable_urls

DONE_KEY = "podcast_done"  # kv list of already-aired topics; pending = TOPICS not yet in it
TOPICS: list[str] = sources_loader.load(Path(__file__).parent, []) or []
MAX_SOURCE_URLS = 10
_MAX_MODEL_ATTEMPTS = 4
_SOURCE_SEPARATOR = "\n\n"
# Transcript: prefer Gemini 3.1 Flash (AI Studio key, free tier), fall back to OpenRouter's free
# models. podcastfy reads each model's key from the env var named by api_key_label.
_TRANSCRIPT_MODEL = "gemini/gemini-3.1-flash"
_GOOGLE_AI_STUDIO_KEY_LABEL = "GOOGLE_AI_STUDIO_KEY"
_OPENROUTER_KEY_LABEL = "OPENROUTER_API_KEY"
# Google Cloud TTS, keyed by GEMINI_API_KEY. Passed as an explicit arg because podcastfy ignores a
# nested text_to_speech override, else defaults to openai.
_TTS_MODEL = "gemini"
DISCOVER_PROMPT = (Path(__file__).parent / "source_discovery_prompt.md").read_text()
CONVERSATION_CONFIG = {
    "conversation_style": ["technical", "narrative", "engaging", "story-driven"],
    "roles_person1": "curious host who keeps one narrative thread going with sharp questions",
    "roles_person2": "expert who explains via cause-and-effect and vivid examples, not lists",
    # Story beats, deliberately with no "Introduction"/"Conclusion"/"Takeaways" beat: longform
    # generates each chunk against this structure, so a conclusion beat makes every section end
    # with a wrap-up and a goodbye.
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
    "creativity": 0.7,  # 0.3 flattened the dialogue into a fact list; higher tracks the narrative
    # Chirp 3 HD voices (GA, more natural) instead of the default Journey pair, read by the gemini
    # Cloud-TTS provider from text_to_speech.<provider>.default_voices.
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
    which topics have aired, so adding, removing, or reordering topics there takes effect
    immediately rather than drifting from a separately-persisted queue."""
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
        return Result(subject=subject, markdown="", artifacts=[], meta={"topic": topic})

    state_mod.set_kv(ctx.state, DONE_KEY, sorted(done | {topic}))  # mark aired only on success
    return Result(subject=subject, markdown="", artifacts=[audio_path], meta={"topic": topic})


# == Source discovery =========================================================


def _discover_urls(ctx: Context, topic: str) -> list[str]:
    """Ask the LLM for candidate source URLs, capped at MAX_SOURCE_URLS."""
    raw = ctx.call(system=DISCOVER_PROMPT, user=topic)
    urls = [line.strip() for line in raw.splitlines() if line.strip().startswith("http")]
    return urls[:MAX_SOURCE_URLS]


# == Episode generation =======================================================


async def _generate_episode(ctx: Context, topic: str) -> str | None:
    """Episode from reachable discovered URLs (or the bare topic); None if every model fails.

    podcastfy drives its own transcript LLM call with a single model and no fallback, so we
    cascade through the resolved candidates here — free models are frequently saturated upstream."""
    urls = await reachable_urls(_discover_urls(ctx, topic))
    if urls:
        ctx.logger.info(f"podcast: {len(urls)} reachable source url(s) for {topic!r}")
    else:
        ctx.logger.warning("⚠️ podcast: no reachable source URLs for %r; using topic text", topic)
    # Extract source text once so a model retry re-runs only the transcript LLM call, not
    # podcastfy's per-attempt headless-browser crawl of every URL.
    source_text = await _extract_sources(urls) if urls else ""
    if urls and not source_text:
        ctx.logger.warning("⚠️ podcast: no text extracted from source URLs; podcastfy will re-crawl")
    pf_urls = None if source_text else (urls or None)
    pf_text = source_text or (None if urls else topic)

    instructions = (Path(__file__).parent / "prompt.md").read_text()
    from podcastfy.client import generate_podcast

    last_err: Exception | None = None
    for model, key_label in _transcript_candidates():
        try:
            return generate_podcast(
                urls=pf_urls,
                text=pf_text,
                conversation_config={**CONVERSATION_CONFIG, "user_instructions": instructions},
                llm_model_name=model,
                api_key_label=key_label,
                tts_model=_TTS_MODEL,
                longform=True,
            )
        except Exception as exc:  # tolerate any generation failure and try the next model
            last_err = exc
            ctx.logger.warning("⚠️ podcast: model=%s failed: %s", model, exc)
    ctx.logger.warning("⚠️ podcast: all transcript models failed for %r: %s", topic, last_err)
    return None


def _transcript_candidates() -> list[tuple[str, str]]:
    """(model, api-key env var) to try in order: Gemini 3.1 Flash first when its AI Studio key is
    set, then the free OpenRouter cascade. podcastfy runs one model with no fallback of its own."""
    candidates: list[tuple[str, str]] = []
    if os.environ.get(_GOOGLE_AI_STUDIO_KEY_LABEL):
        candidates.append((_TRANSCRIPT_MODEL, _GOOGLE_AI_STUDIO_KEY_LABEL))
    free = llm.resolve_models(podcast=True) or [llm.FALLBACK_MODEL]
    candidates += [(m, _OPENROUTER_KEY_LABEL) for m in free]
    return candidates[:_MAX_MODEL_ATTEMPTS]


async def _extract_sources(urls: list[str]) -> str:
    """Extract and join article bodies from the reachable URLs; '' if none yield text."""
    texts = await asyncio.gather(*(asyncio.to_thread(article_text, url) for url in urls))
    return _SOURCE_SEPARATOR.join(text for text in texts if text)
