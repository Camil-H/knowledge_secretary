"""Podcast task: rotate through topics.yaml and generate a long-form two-host
episode via podcastfy, using the free Microsoft Edge TTS backend and the
transcript LLM configured under models.podcast.primary.
"""

from pathlib import Path

import yaml

from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks

TOPICS_KEY = "topics"
STATE_KEY = "podcast_idx"


@tasks.register("podcast")
def run(ctx: Context) -> Result:
    """Advance the topic rotation, generate a podcast episode for the picked
    topic via podcastfy, and return it as a Result with the mp3 (if any) as
    the sole artifact.
    """
    topics = _load_topics(ctx.cfg["tasks"]["podcast"]["topics_file"])
    topic = _advance_topic(ctx.state, topics)
    ctx.log(f"podcast: topic_idx={state_mod.get_kv(ctx.state, STATE_KEY)} topic={topic!r}")

    subject = f"Podcast — {topic}"
    audio_path = _generate_episode(topic, ctx)
    if audio_path is None:
        return Result(subject=subject, markdown="", artifacts=[], meta={"topic": topic})
    return Result(subject=subject, markdown="", artifacts=[audio_path], meta={"topic": topic})


# == Helper Functions ========================================================


def _load_topics(topics_file: str) -> list[str]:
    with open(topics_file) as f:
        data = yaml.safe_load(f)
    return data[TOPICS_KEY]


def _advance_topic(state: dict, topics: list[str]) -> str:
    idx = state_mod.get_kv(state, STATE_KEY, -1)
    nxt = (idx + 1) % len(topics)
    state_mod.set_kv(state, STATE_KEY, nxt)
    return topics[nxt]


def _split_llm_model(model_id: str) -> str:
    """Map a LiteLLM-style "provider/model" id to what podcastfy's
    llm_model_name expects. podcastfy special-cases Gemini by constructing
    ChatGoogleGenerativeAI directly with the bare model name (checking only
    `"gemini" in model_name.lower()`); every other provider is routed through
    ChatLiteLLM, which accepts the "provider/model" form as-is.
    """
    provider, _, model_name = model_id.partition("/")
    if provider.lower() == "gemini" and model_name:
        return model_name
    return model_id


def _generate_episode(topic: str, ctx: Context) -> str | None:
    """Call podcastfy.generate_podcast for a long-form, two-host episode
    seeded by `topic`, guided by this bucket's prompt.md, narrated with the
    free Microsoft Edge TTS voices. Never raises: degrade to None on any
    failure so the feed deliverer can skip this run instead of crashing the
    pipeline.
    """
    instructions = (Path(__file__).parent / "prompt.md").read_text()
    llm_model_name = _split_llm_model(ctx.cfg["models"]["podcast"]["primary"][0])
    try:
        from podcastfy.client import generate_podcast

        return generate_podcast(
            text=topic,
            llm_model_name=llm_model_name,
            tts_model="edge",
            longform=True,
            conversation_config={"user_instructions": instructions},
        )
    except Exception as exc:
        ctx.log(f"podcast: generate_podcast failed: {exc}")
        return None
