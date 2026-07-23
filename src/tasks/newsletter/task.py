"""Newsletter digest (gather-based; see src/tasks/runner.py). One editor pass when the
day's items fit the prompt budget; a map-reduce over batches when they don't — every
source is always represented, just trimmed harder on busy days."""

from datetime import UTC, datetime
from itertools import batched
from pathlib import Path

from src.core import sources_loader
from src.core.models import Context, Item, Result, SourceSpec
from src.core.registry import tasks
from src.tasks.newsletter.utils import clean
from src.tasks.runner import run_source_task

EDITOR_PROMPT = (Path(__file__).parent / "prompt.md").read_text()
SYNTHESIS_PROMPT = (Path(__file__).parent / "synthesis_prompt.md").read_text()
SOURCES: list[SourceSpec] = sources_loader.load(Path(__file__).parent, []) or []
ITEM_CHAR_LIMIT = 20000
# Sized for the smallest model call() might fall back to (~32k tokens), not the selected one —
# we can't know which rung of the cascade answers.
TOTAL_CHAR_BUDGET = 120000
ITEM_CHAR_FLOOR = 1000


# == Task =====================================================================


@tasks.register("newsletter")
def run(ctx: Context) -> Result:
    subject = f"Knowledge Secretary — {datetime.now(UTC):%Y-%m-%d}"
    return run_source_task(ctx, SOURCES, _produce, subject)


# == Produce ==================================================================


def _produce(ctx: Context, items: list[Item]) -> str:
    """Synthesize the newsletter, always including every source.

    Each item gets an equal share of the prompt budget, trimmed harder as volume grows.
    When even the per-item floor won't fit one prompt, fall back to map-reduce so nothing
    is dropped."""
    if ITEM_CHAR_FLOOR * len(items) > TOTAL_CHAR_BUDGET:
        return _map_reduce(ctx, items)
    per_item = _per_item_budget(len(items))
    return ctx.call(system=EDITOR_PROMPT, user=_editor_input(items, per_item))


def _map_reduce(ctx: Context, items: list[Item]) -> str:
    """Summarize items in budget-sized batches, then synthesize one newsletter from the
    batch fragments — every source survives, at the cost of extra LLM calls."""
    batch_size = TOTAL_CHAR_BUDGET // ITEM_CHAR_FLOOR
    fragments = [
        ctx.call(system=EDITOR_PROMPT, user=_editor_input(list(batch), ITEM_CHAR_FLOOR))
        for batch in batched(items, batch_size)
    ]
    return ctx.call(system=SYNTHESIS_PROMPT, user="\n\n".join(fragments))


# == Helper Functions =========================================================


def _per_item_budget(n_items: int) -> int:
    """Per-item body budget: an equal share of TOTAL_CHAR_BUDGET, clamped to
    [ITEM_CHAR_FLOOR, ITEM_CHAR_LIMIT]."""
    return max(ITEM_CHAR_FLOOR, min(TOTAL_CHAR_BUDGET // n_items, ITEM_CHAR_LIMIT))


def _editor_input(items: list[Item], per_item: int) -> str:
    """One block per item (title, url, body trimmed to `per_item`), grouped by section."""
    grouped: dict[str, list[Item]] = {}
    for item in items:
        grouped.setdefault(item.section, []).append(item)

    blocks: list[str] = []
    for section, entries in grouped.items():
        blocks.append(f"## {section}")
        for item in entries:
            body = clean(item.text)[:per_item] or "(no content available)"
            blocks.append(f"### {item.title}\n{item.url}\n{body}")
    return "\n\n".join(blocks)
