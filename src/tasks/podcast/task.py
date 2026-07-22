"""Podcast task: work through the topics in this dir's sources.yaml as a queue.
Each run pops the next topic, asks an LLM for source URLs about it, keeps the
reachable ones, and generates a long-form two-host episode from them via
podcastfy (Gemini transcript + geminimulti multi-speaker TTS). A generated topic
is removed from the queue so it never repeats; an empty queue produces nothing.
The queue is seeded from sources.yaml on the first run and then lives in the
committed state.
"""

import asyncio
from pathlib import Path

import httpx

from src.core import sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks

QUEUE_KEY = "podcast_queue"  # kv list of topics still to do; seeded from TOPICS
TOPICS = sources_loader.load(Path(__file__).parent, [])
MAX_SOURCE_URLS = 5
_URL_CHECK_TIMEOUT_S = 10
_TTS_MODEL = "geminimulti"  # Google multi-speaker TTS (en-US-Studio-MultiSpeaker)
_LLM_MODEL = "gemini-3.0-flash"  # transcript LLM
_GEMINI_KEY_LABEL = "GEMINI_API_KEY"  # env var name podcastfy reads the key from
DISCOVER_PROMPT = (Path(__file__).parent / "discover.md").read_text()
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


@tasks.register("podcast")
def run(ctx: Context) -> Result:
    """Pop the next topic off the queue, generate its episode, and return it as a
    Result with the mp3 (if any) as the sole artifact. The topic is removed from
    the queue once generated; an empty queue produces nothing.
    """
    queue = state_mod.get_kv(ctx.state, QUEUE_KEY, list(TOPICS))
    if not queue:
        ctx.log("podcast: topic queue empty — nothing to generate")
        return Result(subject="Podcast — (queue empty)", markdown="")

    topic = queue[0]
    ctx.log(f"podcast: topic={topic!r} ({len(queue)} left)")
    subject = f"Podcast — {topic}"
    audio_path = asyncio.run(_generate_episode(topic, ctx))
    if audio_path is None:
        return Result(subject=subject, markdown="", artifacts=[], meta={"topic": topic})

    state_mod.set_kv(ctx.state, QUEUE_KEY, queue[1:])  # remove the generated topic
    return Result(subject=subject, markdown="", artifacts=[audio_path], meta={"topic": topic})


# == Helper Functions =========================================================


def _discover_urls(ctx: Context, topic: str) -> list[str]:
    """Ask the LLM for candidate source URLs, capped at MAX_SOURCE_URLS."""
    raw = ctx.call("summarize", system=DISCOVER_PROMPT, user=topic)
    urls = [line.strip() for line in raw.splitlines() if line.strip().startswith("http")]
    return urls[:MAX_SOURCE_URLS]


async def _validate_urls(urls: list[str]) -> list[str]:
    """Keep only the URLs that respond < 400, checked concurrently."""
    if not urls:
        return []
    async with httpx.AsyncClient(timeout=_URL_CHECK_TIMEOUT_S, follow_redirects=True) as client:
        oks = await asyncio.gather(*(_url_ok(client, url) for url in urls))
    return [url for url, ok in zip(urls, oks, strict=True) if ok]


async def _url_ok(client: httpx.AsyncClient, url: str) -> bool:
    try:
        resp = await client.head(url)
        if resp.status_code >= 400:  # some servers reject HEAD — confirm with GET
            resp = await client.get(url)
        return resp.status_code < 400
    except Exception:
        return False


async def _generate_episode(topic: str, ctx: Context) -> str | None:
    """Discover reachable source URLs for `topic` and generate a long-form two-host
    episode from them via podcastfy (falling back to the bare topic if none are
    reachable). Degrades to None on any failure so a bad run never crashes the
    pipeline.
    """
    urls = await _validate_urls(_discover_urls(ctx, topic))
    ctx.log(f"podcast: {len(urls)} reachable source url(s) for {topic!r}")
    instructions = (Path(__file__).parent / "prompt.md").read_text()
    try:
        from podcastfy.client import generate_podcast

        return generate_podcast(
            urls=urls or None,
            text=None if urls else topic,
            conversation_config={**CONVERSATION_CONFIG, "user_instructions": instructions},
            tts_model=_TTS_MODEL,
            llm_model_name=_LLM_MODEL,
            api_key_label=_GEMINI_KEY_LABEL,
            longform=True,
        )
    except Exception as exc:
        ctx.log(f"podcast: generate_podcast failed: {exc}")
        return None
