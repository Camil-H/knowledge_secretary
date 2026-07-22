"""Newsletter digest task.

Two-stage so nothing is judged on a truncated RSS headline:
1. Summarize each new item from its FULL fetched body (sources are enriched with
   article_text upstream; the per-item summarizer works off the full text and can
   flag an item IRRELEVANT for broad sources). Short items (tweets, thin teasers)
   pass through verbatim to save calls.
2. Synthesize the per-item summaries into one editor-written newsletter.

Mirrors the YouTube task's per-item pattern and keeps each LLM call small.
See CONTRACTS.md.
"""

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.core import sources_loader
from src.core.models import Context, Item, Result
from src.core.registry import tasks
from src.tasks.newsletter import adapters as _adapters  # noqa: F401  (registers adapters/enrichers)

SYNTHESIS_PROMPT = (Path(__file__).parent / "prompt.md").read_text()
ITEM_PROMPT = (Path(__file__).parent / "item_prompt.md").read_text()
SOURCES = sources_loader.load(Path(__file__).parent, [])
ITEM_CHAR_LIMIT = 12000  # full body passed to the per-item summarizer
PASSTHROUGH_CHARS = 400  # shorter items (tweets, teasers) skip the per-item LLM call
IRRELEVANT = "IRRELEVANT"  # sentinel the item summarizer returns for off-topic items


# == Task =====================================================================


@tasks.register("newsletter")
def run(ctx: Context) -> Result:
    """Gather new items, summarize each from full content, then synthesize."""
    since = datetime.now(UTC) - timedelta(hours=ctx.cfg.get("window_hours", 24))

    items = ctx.gather(SOURCES, since)
    ctx.log(f"newsletter: gathered {len(items)} new item(s)")

    subject = f"Knowledge Secretary — {datetime.now(UTC):%Y-%m-%d}"
    # all gathered items are consumed (processed) regardless of relevance, so
    # off-topic items aren't re-summarized every day.
    consumed = [it.id for it in items]
    if not items:
        return Result(subject=subject, markdown="")

    relevant = [
        (it, s)
        for it, s in ((it, _item_summary(ctx, it)) for it in items)
        if s.strip().upper() != IRRELEVANT
    ]
    if not relevant:
        ctx.log("newsletter: all items filtered as irrelevant")
        return Result(subject=subject, markdown="", consumed=consumed)

    markdown = ctx.call("summarize", system=SYNTHESIS_PROMPT, user=_synthesis_input(relevant))
    return Result(subject=subject, markdown=markdown, consumed=consumed)


# == Helper Functions =========================================================


def _item_summary(ctx: Context, item: Item) -> str:
    """Summarize one item from its FULL body; short items pass through verbatim."""
    text = _clean(item.text)
    if len(text) < PASSTHROUGH_CHARS:  # tweets / thin teasers: not worth an LLM call
        return text or "(no content available)"
    user = f"Section: {item.section}\nTitle: {item.title}\nURL: {item.url}\nContent:\n{text}"
    return ctx.call("summarize", system=ITEM_PROMPT, user=user)


def _synthesis_input(summarized: list[tuple[Item, str]]) -> str:
    """Group per-item summaries by section for the editor pass."""
    grouped: dict[str, list[tuple[Item, str]]] = {}
    for item, summary in summarized:
        grouped.setdefault(item.section, []).append((item, summary))

    lines: list[str] = []
    for section, entries in grouped.items():
        lines.append(f"## {section}")
        for item, summary in entries:
            lines.append(f"- {item.title} ({item.url})")
            lines.append(f"  {summary}")
    return "\n".join(lines)


def _clean(text: str) -> str:
    """Collapse whitespace and cap at the per-item body budget."""
    return re.sub(r"\s+", " ", text or "").strip()[:ITEM_CHAR_LIMIT]
