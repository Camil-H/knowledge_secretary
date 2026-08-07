"""YouTube digest (gather-based; see src/tasks/runner.py): summarize the new videos'
transcripts in batches, render grouped by section."""

import re
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path

from src import config
from src.core import sources_loader
from src.core.models import Context, Item, Result, SourceSpec
from src.core.registry import tasks
from src.tasks.runner import run_source_task

PROMPT = (Path(__file__).parent / "prompt.md").read_text()
_NO_TRANSCRIPT = ["- (no transcript available)"]
_NO_SUMMARY = ["- (summary unavailable)"]
_WATCH_HEADING = "To watch"
_BLOCK_TAG = "VIDEO"
# tolerates decoration the model may add around the header (`**[VIDEO yt:x]**`, `## VIDEO yt:x`)
_BLOCK_HEADER = re.compile(rf"^\W*{_BLOCK_TAG}\s+([^\]\s]+)")
SOURCES: list[SourceSpec] = sources_loader.load(Path(__file__).parent, []) or []


# == Task =====================================================================


@tasks.register("youtube")
def run(ctx: Context) -> Result:
    subject = f"YouTube Digest — {datetime.now(UTC):%Y-%m-%d}"
    return run_source_task(ctx, SOURCES, _produce, subject)


# == Produce ==================================================================


def _produce(ctx: Context, items: list[Item]) -> str:
    """Summarize the new videos, grouped by section (config order), then list the
    watch-only ones unsummarized under their own trailing heading.

    A video the batch summary omits or garbles degrades to a note on its own; the rest of
    its batch is unaffected."""
    watch = [item for item in items if item.meta.get("watch")]
    summarized = [item for item in items if not item.meta.get("watch")]
    bullets = _batched_summaries(ctx, [item for item in summarized if item.text])
    grouped: dict[str, list[tuple[Item, list[str]]]] = {}
    for item in summarized:
        lines = (bullets.get(item.id) or _NO_SUMMARY) if item.text else _NO_TRANSCRIPT
        grouped.setdefault(item.section, []).append((item, lines))

    if watch:
        ctx.logger.info(f"youtube: {len(watch)}/{len(items)} videos listed to watch, unsummarized")
    missing = sum(1 for it in summarized if not it.text)
    if missing:
        ctx.logger.info(f"youtube: {missing}/{len(summarized)} videos had no transcript")
    unsummarized = sum(1 for it in summarized if it.text and not bullets.get(it.id))
    if unsummarized:
        ctx.logger.warning(
            f"⚠️ youtube: {unsummarized}/{len(summarized)} videos absent from the batch summaries"
        )
    return _render(_section_order(SOURCES), grouped, _newest_first(watch))


# == Summaries ================================================================


def _batched_summaries(ctx: Context, items: list[Item]) -> dict[str, list[str]]:
    """Bullet lines per video id, one model call per `config.YOUTUBE_BATCH_SIZE` videos.

    Videos the model drops or mis-labels are simply absent from the mapping — the caller
    decides what that means for them."""
    summaries: dict[str, list[str]] = {}
    for batch in batched(items, config.YOUTUBE_BATCH_SIZE):
        raw = ctx.call(system=PROMPT, user=_batch_input(batch))
        summaries.update(_parse_blocks(raw, {item.id for item in batch}))
    return summaries


def _batch_input(items: tuple[Item, ...]) -> str:
    """One id-headed block per video, each transcript trimmed to the per-video char limit."""
    return "\n\n".join(
        f"[{_BLOCK_TAG} {item.id}]\n"
        f"Title: {item.title}\n"
        f"Channel: {item.meta.get('channel', '')}\n"
        f"Transcript:\n{item.text[: config.YOUTUBE_TRANSCRIPT_CHAR_LIMIT]}"
        for item in items
    )


def _parse_blocks(raw: str, requested_ids: set[str]) -> dict[str, list[str]]:
    """Bullet lines per video id, split on the reply's `[VIDEO <id>]` headers.

    A header whose id wasn't requested closes the current block without opening a new one, so
    an invented or mangled id costs that one video rather than pinning its bullets on another."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in raw.splitlines():
        header = _BLOCK_HEADER.match(line)
        if header:
            current = header[1] if header[1] in requested_ids else None
        elif current and line.strip():
            blocks.setdefault(current, []).append(line.strip())
    return blocks


# == Render ===================================================================


def _section_order(specs: list[SourceSpec]) -> list[str]:
    """Section names in the order they first appear across the task's specs."""
    order: list[str] = []
    for spec in specs:
        if spec["section"] not in order:
            order.append(spec["section"])
    return order


def _newest_first(items: list[Item]) -> list[Item]:
    """Watch entries read as a release radar, so recency wins over the spec order the
    summarized sections keep."""
    return sorted(items, key=lambda item: item.published, reverse=True)


def _render(
    section_order: list[str],
    grouped: dict[str, list[tuple[Item, list[str]]]],
    watch: list[Item],
) -> str:
    """Render the digest markdown: sections in config order, then the flat watch list."""
    lines: list[str] = []
    for section in section_order:
        entries = grouped.get(section) or []
        if not entries:
            continue
        lines.append(f"- {section}")
        for item, bullets in entries:
            lines.append(_entry_line(item))
            lines.extend(f"        {bullet}" for bullet in bullets)
    if watch:
        lines.append(f"- {_WATCH_HEADING}")
        lines.extend(_entry_line(item) for item in watch)
    return "\n".join(lines)


def _entry_line(item: Item) -> str:
    return f"    - [{item.title}]({item.url}) -- {item.meta.get('channel', '')}"
