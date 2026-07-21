"""Newsletter digest task.

Groups newly gathered blog/paper/preprint/social items by section and makes a
single LLM call to synthesize them into one newsletter. See CONTRACTS.md.
"""

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.core.models import Context, Item, Result
from src.core.registry import tasks

PROMPT = (Path(__file__).parent / "prompt.md").read_text()
TEXT_CHAR_LIMIT = 2000  # per-item text budget so the synthesis call stays cheap


# == Task =====================================================================


@tasks.register("newsletter")
def run(ctx: Context) -> Result:
    """Gather new items across the task's sources and synthesize one digest."""
    task_cfg = ctx.cfg["tasks"]["newsletter"]
    since = datetime.now(UTC) - timedelta(hours=ctx.cfg.get("window_hours", 24))

    items = ctx.gather(task_cfg["sources"], since)
    ctx.log(f"newsletter: gathered {len(items)} new item(s)")

    subject = f"Knowledge Secretary — {datetime.now(UTC):%Y-%m-%d}"
    if not items:
        return Result(subject=subject, markdown="")

    summary_md = ctx.call("summarize", system=PROMPT, user=_build_user_message(items))
    # consumed reported (not marked here) so the dispatcher only burns these ids
    # once the email is actually sent.
    return Result(subject=subject, markdown=summary_md, consumed=[it.id for it in items])


# == Helper Functions =========================================================


def _build_user_message(items: list[Item]) -> str:
    """Group items by section and render a compact block per item for the model."""
    grouped: dict[str, list[Item]] = {}
    for item in items:
        grouped.setdefault(item.section, []).append(item)

    lines: list[str] = []
    for section, section_items in grouped.items():
        lines.append(f"## {section}")
        for item in section_items:
            lines.append(f"- Title: {item.title}")
            lines.append(f"  URL: {item.url}")
            lines.append(f"  Text: {_clean(item.text)}")
    return "\n".join(lines)


def _clean(text: str) -> str:
    """Collapse whitespace and truncate to the per-item char budget."""
    return re.sub(r"\s+", " ", text or "").strip()[:TEXT_CHAR_LIMIT]
