"""Podcast task: generate a two-host episode for the next unaired topic — research, the
longform transcript and the Cloud TTS orchestration all in-repo."""

import os
import tempfile
from pathlib import Path

from src import config
from src.clients import gemini, tavily
from src.core import ledger as ledger_mod
from src.core import sources_loader
from src.core import state as state_mod
from src.core.models import Context, Result
from src.core.registry import tasks
from src.tasks.podcast import audio, transcript

DONE_KEY = "podcast_done"
TOPICS: list[str] = sources_loader.load(Path(__file__).parent, []) or []
RESEARCH_PROMPT = (Path(__file__).parent / "research_prompt.md").read_text()
EPISODE_FILENAME = "episode.mp3"
NO_EPISODE_NOTICE = (
    "No episode today — research or generation failed. The topic stays queued and is retried "
    "on the next run."
)


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

    state_mod.set_kv(ctx.state, DONE_KEY, sorted(done | {topic}))
    return Result(subject=subject, markdown="", artifacts=[audio_path], meta={"topic": topic})


# == Episode generation =======================================================


def _generate_episode(ctx: Context, topic: str) -> str | None:
    """Episode audio for the topic; None if any stage fails."""
    overview = _research(ctx, topic)
    if not overview:
        return None
    ctx.logger.info("podcast: %d chars of research for %r", len(overview), topic)

    try:
        text = transcript.generate(topic, overview, call=ctx.call)
    except Exception as exc:  # a queued topic is retried next run; a broken episode is not
        ctx.logger.warning("⚠️ podcast: transcript failed for %r: %s", topic, exc)
        return None
    ctx.logger.info("podcast: %d-char transcript for %r", len(text), topic)
    return _synthesize(ctx, text, topic)


# == Content research =========================================================


def _research(ctx: Context, topic: str) -> str:
    """Source material for the topic, written up from searched pages; "" when unavailable.

    The source URLs are logged here because this is where an episode's provenance is decided: a
    bad episode has to be traceable to what it was built from."""
    try:
        pages = tavily.search(topic)
    except Exception as exc:
        ctx.logger.warning("⚠️ podcast: search failed for %r: %s", topic, exc)
        return ""
    if len(pages) < config.TAVILY_MIN_RESULTS:
        ctx.logger.warning(
            "⚠️ podcast: only %d usable source(s) for %r, need %d",
            len(pages),
            topic,
            config.TAVILY_MIN_RESULTS,
        )
        return ""
    ctx.logger.info(
        "podcast: %d source(s) for %r: %s", len(pages), topic, ", ".join(p["url"] for p in pages)
    )
    try:
        return gemini.call(RESEARCH_PROMPT, _research_input(topic, pages), ledger=ledger_mod.load())
    except Exception as exc:
        ctx.logger.warning("⚠️ podcast: research failed for %r: %s", topic, exc)
        return ""


def _research_input(topic: str, pages: list[dict[str, str]]) -> str:
    """The topic and its searched pages as one prompt input.

    The topic leads and only the sources are truncated: the prompt's instruction to stay on topic
    is worthless if the budget can cut the topic itself off the end."""
    blocks = [f"## {p['title']}\n{p['url']}\n\n{p['text']}" for p in pages]
    sources = "\n\n".join(blocks)[: config.TAVILY_MAX_SOURCES_CHARS]
    return f"# Topic\n{topic}\n\n# Sources\n{sources}"


# == Audio synthesis ==========================================================


def _synthesize(ctx: Context, text: str, topic: str) -> str | None:
    """Episode audio for the transcript at a fresh temp path; None on failure."""
    out_path = os.path.join(tempfile.mkdtemp(), EPISODE_FILENAME)
    try:
        return audio.synthesize(text, out_path, ledger=ledger_mod.load())
    except Exception as exc:
        ctx.logger.warning("⚠️ podcast: audio synthesis failed for %r: %s", topic, exc)
        return None
