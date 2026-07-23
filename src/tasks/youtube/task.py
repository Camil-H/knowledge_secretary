"""YouTube digest (gather-based; see src/tasks/runner.py): summarize each new
video's transcript, render grouped by section."""

from datetime import UTC, datetime
from pathlib import Path

from src.core import sources_loader
from src.core.models import Context, Item, Result, SourceSpec
from src.core.registry import tasks
from src.tasks.runner import run_source_task

PROMPT = (Path(__file__).parent / "prompt.md").read_text()
TRANSCRIPT_CHAR_LIMIT = 12000
_NO_TRANSCRIPT = ["- (no transcript available)"]
SOURCES: list[SourceSpec] = sources_loader.load(Path(__file__).parent, []) or []


# == Task =====================================================================


@tasks.register("youtube")
def run(ctx: Context) -> Result:
    subject = f"YouTube Digest — {datetime.now(UTC):%Y-%m-%d}"
    return run_source_task(ctx, SOURCES, _produce, subject)


# == Produce ==================================================================


def _produce(ctx: Context, items: list[Item]) -> str:
    """Summarize each new video's transcript, grouped by section (config order)."""
    grouped: dict[str, list[tuple[Item, list[str]]]] = {}
    for item in items:
        grouped.setdefault(item.section, []).append((item, _summarize(ctx, item)))
    missing = sum(1 for it in items if not it.text)
    if missing:
        ctx.logger.info(f"youtube: {missing}/{len(items)} videos had no transcript")
    return _render(_section_order(SOURCES), grouped)


def _section_order(specs: list[SourceSpec]) -> list[str]:
    """Section names in the order they first appear across the task's specs."""
    order: list[str] = []
    for spec in specs:
        if spec["section"] not in order:
            order.append(spec["section"])
    return order


def _summarize(ctx: Context, item: Item) -> list[str]:
    """The model's bullet lines for one video, or a note if there's no transcript."""
    if not item.text:  # the transcript fetcher already logged why it's empty
        return _NO_TRANSCRIPT
    user = (
        f"Title: {item.title}\n"
        f"Channel: {item.meta.get('channel', '')}\n"
        f"Transcript:\n{item.text[:TRANSCRIPT_CHAR_LIMIT]}"
    )
    raw = ctx.call(system=PROMPT, user=user)
    return [line for line in raw.splitlines() if line.strip()]


def _render(section_order: list[str], grouped: dict[str, list[tuple[Item, list[str]]]]) -> str:
    """Render the digest markdown, grouped by section in config order."""
    lines: list[str] = []
    for section in section_order:
        entries = grouped.get(section) or []
        if not entries:
            continue
        lines.append(f"- {section}")
        for item, bullets in entries:
            lines.append(f"    - [{item.title}]({item.url}) -- {item.meta.get('channel', '')}")
            lines.extend(f"        {bullet}" for bullet in bullets)
    return "\n".join(lines)
