"""Podcast task: pop the next topic from the sources.yaml queue and generate a
two-host episode via podcastfy (free OpenRouter transcript, free Edge TTS)."""

import asyncio
from pathlib import Path

from src.core import llm, sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks
from src.tasks.podcast.utils import validate_urls

QUEUE_KEY = "podcast_queue"  # kv list of topics still to do; seeded from TOPICS
TOPICS = sources_loader.load(Path(__file__).parent, [])
MAX_SOURCE_URLS = 10
_OPENROUTER_KEY_LABEL = "OPENROUTER_API_KEY"  # transcript LLM: podcastfy -> LiteLLM -> OpenRouter
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
    # free Edge TTS (no key); podcastfy's own default is the paid openai voice
    "text_to_speech": {"default_tts_model": "edge"},
}


# == Task =====================================================================


@tasks.register("podcast")
def run(ctx: Context) -> Result:
    """Pop the next queued topic, generate its episode, drop it from the queue on success."""
    queue = state_mod.get_kv(ctx.state, QUEUE_KEY, list(TOPICS))
    if not queue:
        ctx.log("podcast: topic queue empty — nothing to generate")
        return Result(subject="Podcast — (queue empty)", markdown="")

    topic = queue[0]
    ctx.log(f"podcast: topic={topic!r} ({len(queue)} left)")
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
    """Episode from reachable discovered URLs (or the bare topic); None on any failure."""
    urls = await validate_urls(_discover_urls(ctx, topic))
    ctx.log(f"podcast: {len(urls)} reachable source url(s) for {topic!r}")
    instructions = (Path(__file__).parent / "prompt.md").read_text()
    model = (llm.resolve_models(podcast=True) or [llm.FALLBACK_MODEL])[0]
    try:
        from podcastfy.client import generate_podcast

        return generate_podcast(
            urls=urls or None,
            text=None if urls else topic,
            conversation_config={**CONVERSATION_CONFIG, "user_instructions": instructions},
            llm_model_name=model,
            api_key_label=_OPENROUTER_KEY_LABEL,
            longform=True,
        )
    except Exception as exc:
        ctx.log(f"podcast: generate_podcast failed: {exc}")
        return None
