"""Podcast task: work through the topics in this dir's sources.yaml as a queue.
Each run pops the next topic and generates a long-form two-host episode via
podcastfy (free Microsoft Edge TTS, best free OpenRouter model for the
transcript). A generated topic is removed from the queue, so it never repeats;
an empty queue produces nothing. The queue is seeded from sources.yaml on the
first run and then lives in the committed state.
"""

from pathlib import Path

from src.core import llm, sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks

QUEUE_KEY = "podcast_queue"  # kv list of topics still to do; seeded from TOPICS
TOPICS = sources_loader.load(Path(__file__).parent, [])
_OPENROUTER_KEY_LABEL = "OPENROUTER_API_KEY"
# used only if the live free-model list is empty; harmless if stale (podcastfy just fails softly)
_PODCAST_FALLBACK_MODEL = "openrouter/deepseek/deepseek-chat-v3-0324:free"


@tasks.register("podcast")
def run(ctx: Context) -> Result:
    """Pop the next topic off the queue, generate its episode via podcastfy, and
    return it as a Result with the mp3 (if any) as the sole artifact. The topic
    is removed from the queue once generated; an empty queue produces nothing.
    """
    queue = state_mod.get_kv(ctx.state, QUEUE_KEY, list(TOPICS))
    if not queue:
        ctx.log("podcast: topic queue empty — nothing to generate")
        return Result(subject="Podcast — (queue empty)", markdown="")

    topic = queue[0]
    ctx.log(f"podcast: topic={topic!r} ({len(queue)} left)")
    subject = f"Podcast — {topic}"
    audio_path = _generate_episode(topic, ctx)
    if audio_path is None:
        return Result(subject=subject, markdown="", artifacts=[], meta={"topic": topic})

    state_mod.set_kv(ctx.state, QUEUE_KEY, queue[1:])  # remove the generated topic
    return Result(subject=subject, markdown="", artifacts=[audio_path], meta={"topic": topic})


# == Helper Functions =========================================================


def _podcast_model() -> str:
    """Best free OpenRouter model for podcastfy, skipping gemini-named ids.

    podcastfy routes any model whose name contains 'gemini' through its direct
    Google client (needs GEMINI_API_KEY, which we don't set), so pick the top
    non-gemini free model; fall back to a known id if the live list is empty.
    """
    for model in llm.resolve_models("podcast"):
        if "gemini" not in model.lower():
            return model
    return _PODCAST_FALLBACK_MODEL


def _generate_episode(topic: str, ctx: Context) -> str | None:
    """Call podcastfy.generate_podcast for a long-form, two-host episode seeded by
    `topic`, guided by this bucket's prompt.md, narrated with free Edge TTS voices.
    Never raises: degrade to None on any failure so the feed deliverer can skip
    this run instead of crashing the pipeline.
    """
    instructions = (Path(__file__).parent / "prompt.md").read_text()
    model = _podcast_model()
    try:
        from podcastfy.client import generate_podcast

        return generate_podcast(
            text=topic,
            llm_model_name=model,
            api_key_label=_OPENROUTER_KEY_LABEL,
            tts_model="edge",
            longform=True,
            conversation_config={"user_instructions": instructions},
        )
    except Exception as exc:
        ctx.log(f"podcast: generate_podcast failed: {exc}")
        return None
