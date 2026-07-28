"""Podcast task: generate a two-host episode for the next unaired topic — grounded research
and the longform transcript in-repo, podcastfy for audio synthesis only."""

import tempfile
from pathlib import Path

from src.core import sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks
from src.tasks.podcast import content_generator, transcript

DONE_KEY = "podcast_done"
TOPICS: list[str] = sources_loader.load(Path(__file__).parent, []) or []
_TTS_MODEL = "gemini"
NO_EPISODE_NOTICE = (
    "No episode today — grounded research or generation failed. The topic stays queued and is "
    "retried on the next run."
)
_TTS_CONFIG = {
    "text_to_speech": {
        # podcastfy speaks its default "See You Next Time!" as an extra final turn; ours already
        # signs off, so the appended turn must be empty
        "ending_message": "",
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
    try:
        return content_generator.research(topic)
    except Exception as exc:  # no episode is better than one built on nothing
        ctx.logger.warning("⚠️ podcast: research failed for %r: %s", topic, exc)
        return ""


# == Episode generation =======================================================


def _generate_episode(ctx: Context, topic: str) -> str | None:
    """Episode audio from grounded research on the topic; None if any stage fails."""
    overview = _research(ctx, topic)
    if not overview:
        return None
    ctx.logger.info("podcast: %d chars of grounded material for %r", len(overview), topic)

    try:
        text = transcript.generate(topic, overview, call=ctx.call)
    except Exception as exc:  # a queued topic is retried next run; a broken episode is not
        ctx.logger.warning("⚠️ podcast: transcript failed for %r: %s", topic, exc)
        return None
    ctx.logger.info("podcast: %d-char transcript for %r", len(text), topic)
    return _synthesize(ctx, text, topic)


def _synthesize(ctx: Context, text: str, topic: str) -> str | None:
    """Audio for the transcript via podcastfy's transcript-only path; None on failure."""
    # imported here, not at module scope, so the task stays importable without podcastfy
    from podcastfy.client import generate_podcast

    path = _transcript_file(text)
    try:
        return generate_podcast(
            transcript_file=path, tts_model=_TTS_MODEL, conversation_config=_TTS_CONFIG
        )
    except Exception as exc:  # audio is the last stage: nothing left to fall back to
        ctx.logger.warning("⚠️ podcast: audio synthesis failed for %r: %s", topic, exc)
        return None


def _transcript_file(text: str) -> str:
    """The transcript on disk — podcastfy's TTS-only entry point takes a path, not a string."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(text)
    return handle.name
