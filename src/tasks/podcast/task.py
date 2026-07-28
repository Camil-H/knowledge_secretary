"""Podcast task: generate a two-host episode for the next unaired topic via podcastfy
(Gemini 3.1 Flash transcript with an OpenRouter fallback; Google Cloud TTS for audio)."""

import os
from pathlib import Path

from src.core import llm, sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks
from src.tasks.podcast import content_generator

DONE_KEY = "podcast_done"
TOPICS: list[str] = sources_loader.load(Path(__file__).parent, []) or []
MAX_SOURCE_CHARS = 12000  # bounds the transcript input, which drives podcastfy's part count
_MAX_MODEL_ATTEMPTS = 4
_SOURCE_SEPARATOR = "\n\n"
_TRANSCRIPT_MODEL = "gemini/gemini-3.1-flash"
_GOOGLE_AI_STUDIO_KEY_LABEL = "GOOGLE_AI_STUDIO_KEY"
_OPENROUTER_KEY_LABEL = "OPENROUTER_API_KEY"
_TTS_MODEL = "gemini"
INSTRUCTIONS = (Path(__file__).parent / "prompt.md").read_text()
NO_EPISODE_NOTICE = (
    "No episode today — grounded research or generation failed. The topic stays queued and is "
    "retried on the next run."
)
# Episode length: podcastfy instructs each longform part to fill max_output_tokens, so the cap
# is the length lever. ~1200 tokens x ~9 parts ≈ 7k words ≈ 45 min; its default 8192 gave 3h17.
_MAX_OUTPUT_TOKENS = 1200
_MAX_NUM_CHUNKS = 8
_MIN_CHUNK_SIZE = 600
_TTS_FAILURE_MARKERS = (
    "failed to generate audio",
    "error converting text to speech",
    "input.text",
    "input.ssml",
)
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
    audio_path = _generate_episode(ctx, topic)
    if audio_path is None:
        # a notice, not an empty Result: an empty one records nothing at all
        return Result(
            subject=subject, markdown="", meta={"topic": topic}, notices=[NO_EPISODE_NOTICE]
        )

    state_mod.set_kv(ctx.state, DONE_KEY, sorted(done | {topic}))  # mark aired only on success
    return Result(subject=subject, markdown="", artifacts=[audio_path], meta={"topic": topic})


# == Research =================================================================


def _research(ctx: Context, topic: str) -> str:
    """Search-grounded source material for the topic; "" when research is unavailable."""
    api_key = os.environ.get(_GOOGLE_AI_STUDIO_KEY_LABEL)
    if not api_key:
        ctx.logger.warning("⚠️ podcast: %s unset; cannot research", _GOOGLE_AI_STUDIO_KEY_LABEL)
        return ""
    try:
        return content_generator.research(topic, api_key=api_key)
    except Exception as exc:  # no episode is better than one built on nothing
        ctx.logger.warning("⚠️ podcast: research failed for %r: %s", topic, exc)
        return ""


# == Episode generation =======================================================


def _generate_episode(ctx: Context, topic: str) -> str | None:
    """Episode from grounded research on the topic; None if research or every model fails."""
    overview = _research(ctx, topic)
    if not overview:
        return None
    ctx.logger.info("podcast: %d chars of grounded material for %r", len(overview), topic)

    # imported here, not at module scope, so the task stays importable without podcastfy
    from podcastfy.client import generate_podcast

    config = _podcastfy_config()
    last_err: Exception | None = None
    for model, key_label in _transcript_candidates():
        try:
            return generate_podcast(
                urls=None,  # the research text is the whole input; podcastfy must not re-crawl
                text=_episode_text(topic, overview),
                config=config,
                conversation_config={**CONVERSATION_CONFIG, "user_instructions": INSTRUCTIONS},
                llm_model_name=model,
                api_key_label=key_label,
                tts_model=_TTS_MODEL,
                longform=True,
            )
        except Exception as exc:
            last_err = exc
            if _is_tts_failure(exc):
                ctx.logger.warning("⚠️ podcast: audio synthesis failed, not retrying: %s", exc)
                break
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


def _episode_text(topic: str, overview: str) -> str:
    """The transcript input: topic first so the episode stays anchored to it, then the research."""
    return f"{topic}{_SOURCE_SEPARATOR}{overview[:MAX_SOURCE_CHARS]}"


def _podcastfy_config():
    """podcastfy's Config with the per-part output cap lowered from its 8192 default."""
    from podcastfy.utils.config import load_config

    config = load_config()
    generator = {**config.get("content_generator", {}), "max_output_tokens": _MAX_OUTPUT_TOKENS}
    config.configure(content_generator=generator)
    return config


def _is_tts_failure(exc: Exception) -> bool:
    """True for an audio-layer failure, which another transcript model cannot fix."""
    message = str(exc).lower()
    return any(marker in message for marker in _TTS_FAILURE_MARKERS)
