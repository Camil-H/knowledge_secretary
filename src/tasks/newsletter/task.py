"""Newsletter digest (gather-based; see src/tasks/runner.py). One editor pass:
each item's section/title/url/trimmed-body -> prompt.md filters and synthesizes."""

from datetime import UTC, datetime
from pathlib import Path

from src.core import sources_loader
from src.core.models import Context, Item, Result
from src.core.registry import tasks
from src.tasks.newsletter.utils import clean
from src.tasks.runner import run_source_task

EDITOR_PROMPT = (Path(__file__).parent / "prompt.md").read_text()
SOURCES = sources_loader.load(Path(__file__).parent, [])
ITEM_CHAR_LIMIT = 2000  # per-item body budget in the editor input


# == Task =====================================================================


@tasks.register("newsletter")
def run(ctx: Context) -> Result:
    subject = f"Knowledge Secretary — {datetime.now(UTC):%Y-%m-%d}"
    return run_source_task(ctx, SOURCES, _produce, subject)


# == Produce ==================================================================


def _produce(ctx: Context, items: list[Item]) -> str:
    """Render all items into one editor input and synthesize the newsletter."""
    return ctx.call(system=EDITOR_PROMPT, user=_editor_input(items))


def _editor_input(items: list[Item]) -> str:
    """One block per item (title, url, trimmed body), grouped by section."""
    grouped: dict[str, list[Item]] = {}
    for item in items:
        grouped.setdefault(item.section, []).append(item)

    blocks: list[str] = []
    for section, entries in grouped.items():
        blocks.append(f"## {section}")
        for item in entries:
            body = clean(item.text)[:ITEM_CHAR_LIMIT] or "(no content available)"
            blocks.append(f"### {item.title}\n{item.url}\n{body}")
    return "\n\n".join(blocks)
