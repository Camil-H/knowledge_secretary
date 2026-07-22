"""Shared runner for the gather-based tasks (newsletter, youtube).

Both do the same shell — load the task's sources, gather NEW items, hand them to a
task-specific `produce` step, and consume all gathered items. Dedup (state.is_new,
inside gather) does the real "new since the last run" filtering; LOOKBACK_HOURS
just bounds how far back gather scans the feeds (a generous margin so a missed
daily run isn't dropped). The podcast task does NOT use this — it has no gather
step (it rotates a local topic list).
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from src.core.models import Context, Result

LOOKBACK_HOURS = 48  # feed-scan window; dedup filters already-seen items on top


def run_source_task(ctx: Context, sources: list[dict], produce: Callable, subject: str) -> Result:
    """Gather new items for `sources`, render them via `produce` -> markdown, and
    consume all gathered items."""
    since = datetime.now(UTC) - timedelta(hours=LOOKBACK_HOURS)
    items = ctx.gather(sources, since)
    ctx.log(f"{subject}: {len(items)} new item(s)")
    markdown = produce(ctx, items) if items else ""
    return Result(subject=subject, markdown=markdown, consumed=[it.id for it in items])
