"""Podcast task: rotate through the topics in this dir's sources.yaml and generate
a long-form two-host episode via podcastfy, using the free Microsoft Edge TTS
backend and the best free OpenRouter model for the transcript.
"""

from pathlib import Path

from src.core import llm, sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks

STATE_KEY = "podcast_idx"
TOPICS = sources_loader.load(Path(__file__).parent, [])
_OPENROUTER_KEY_LABEL = "OPENROUTER_API_KEY"
# used only if the live free-model list is empty; harmless if stale (podcastfy just fails softly)
_PODCAST_FALLBACK_MODEL = "openrouter/deepseek/deepseek-chat-v3-0324:free"


@tasks.register("podcast")
def run(ctx: Context) -> Result:
    """Advance the topic rotation, generate a podcast episode for the picked
    topic via podcastfy, and return it as a Result with the mp3 (if any) as
    the sole artifact.
    """
    topic = _advance_topic(ctx.state, TOPICS)
    ctx.log(f"podcast: topic_idx={state_mod.get_kv(ctx.state, STATE_KEY)} topic={topic!r}")

    subject = f"Podcast — {topic}"
    audio_path = _generate_episode(topic, ctx)
    if audio_path is None:
        return Result(subject=subject, markdown="", artifacts=[], meta={"topic": topic})
    return Result(subject=subject, markdown="", artifacts=[audio_path], meta={"topic": topic})


# == Helper Functions =========================================================


def _advance_topic(state: dict, topics: list[str]) -> str:
    idx = state_mod.get_kv(state, STATE_KEY, -1)
    nxt = (idx + 1) % len(topics)
    state_mod.set_kv(state, STATE_KEY, nxt)
    return topics[nxt]


def _podcast_model(cfg: dict) -> str:
    """Best free OpenRouter model for podcastfy, skipping gemini-named ids.

    podcastfy routes any model whose name contains 'gemini' through its direct
    Google client (needs GEMINI_API_KEY, which we don't set), so pick the top
    non-gemini free model; fall back to a known id if the live list is empty.
    """
    for model in llm.resolve_models("podcast", cfg):
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
    model = _podcast_model(ctx.cfg)
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
