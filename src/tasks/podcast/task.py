"""Podcast task: pop the next topic from the sources.yaml queue and generate a
two-host episode via podcastfy (OpenRouter transcript LLM, Gemini/Google-Cloud TTS)."""

import asyncio
from pathlib import Path

from src.core import llm, sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks
from src.fetchers.url import article_text
from src.tasks.podcast.utils import validate_urls

QUEUE_KEY = "podcast_queue"  # kv list of topics still to do; seeded from TOPICS
TOPICS = sources_loader.load(Path(__file__).parent, [])
MAX_SOURCE_URLS = 10
_MAX_MODEL_ATTEMPTS = 4  # cap the transcript-LLM fallback cascade
_SOURCE_SEPARATOR = "\n\n"  # joins extracted article bodies into one text blob for podcastfy
_OPENROUTER_KEY_LABEL = "OPENROUTER_API_KEY"  # transcript LLM: podcastfy -> LiteLLM -> OpenRouter
# Google Cloud TTS, keyed by GEMINI_API_KEY (a GCP Cloud-TTS key, not AI Studio). Passed as an
# explicit arg because podcastfy ignores a nested text_to_speech override, else defaults to openai.
_TTS_MODEL = "gemini"
DISCOVER_PROMPT = (Path(__file__).parent / "source_discovery_prompt.md").read_text()
CONVERSATION_CONFIG = {
    "conversation_style": ["technical", "analytical", "engaging"],
    "roles_person1": "curious host who drives the narrative with sharp questions",
    "roles_person2": "domain expert who explains with depth and precision",
    "dialogue_structure": [
        "Introduction",
        "Fundamentals",
        "Mechanisms and Tradeoffs",
        "Edge Cases and Open Questions",
        "Key Takeaways",
    ],
    "podcast_name": "Daily Podcast",
    "podcast_tagline": "A daily podcast",
    "output_language": "English",
    "engagement_techniques": ["analogies", "worked examples", "rhetorical questions"],
    "creativity": 0.3,
}


# == Task =====================================================================


@tasks.register("podcast")
def run(ctx: Context) -> Result:
    """Pop the next queued topic, generate its episode, drop it from the queue on success."""
    queue = state_mod.get_kv(ctx.state, QUEUE_KEY, list(TOPICS))
    if not queue:
        ctx.logger.info("podcast: topic queue empty — nothing to generate")
        return Result(subject="Podcast — (queue empty)", markdown="")

    topic = queue[0]
    ctx.logger.info(f"podcast: topic={topic!r} ({len(queue)} left)")
    subject = f"Podcast — {topic}"
    audio_path = asyncio.run(_generate_episode(ctx, topic))
    if audio_path is None:
        return Result(subject=subject, markdown="", artifacts=[], meta={"topic": topic})

    state_mod.set_kv(ctx.state, QUEUE_KEY, queue[1:])  # remove the generated topic
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
    urls = await validate_urls(_discover_urls(ctx, topic))
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

    models = llm.resolve_models(podcast=True) or [llm.FALLBACK_MODEL]
    last_err: Exception | None = None
    for model in models[:_MAX_MODEL_ATTEMPTS]:
        try:
            return generate_podcast(
                urls=pf_urls,
                text=pf_text,
                conversation_config={**CONVERSATION_CONFIG, "user_instructions": instructions},
                llm_model_name=model,
                api_key_label=_OPENROUTER_KEY_LABEL,
                tts_model=_TTS_MODEL,
                longform=True,
            )
        except Exception as exc:  # tolerate any generation failure and try the next model
            last_err = exc
            ctx.logger.warning("⚠️ podcast: model=%s failed: %s", model, exc)
    ctx.logger.warning("⚠️ podcast: all models failed for %r: %s", topic, last_err)
    return None


async def _extract_sources(urls: list[str]) -> str:
    """Extract and join article bodies from the reachable URLs; '' if none yield text."""
    texts = await asyncio.gather(*(asyncio.to_thread(article_text, url) for url in urls))
    return _SOURCE_SEPARATOR.join(text for text in texts if text)
