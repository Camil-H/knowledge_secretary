"""Fetch driver + shared runner for the gather-based tasks (newsletter, youtube).

`gather` is the single fetch driver each task reaches via `Context.gather`: it
dispatches each source spec to its registered adapter (by `kind`), keeps only NEW
items (dedup) published within the lookback, and runs each spec's enrichers. It
never marks items seen — run.py marks only what a task actually consumes.

`run_source_task` is the shell both gather-based tasks share: load sources,
gather, hand the items to a task-specific `produce`, and consume them all. Dedup
(state.is_new, inside gather) does the real "new since the last run" filtering;
LOOKBACK_HOURS just bounds how far back gather scans the feeds (a generous margin
so a missed daily run isn't dropped). The podcast task uses neither — it has no
gather step (it consumes a local topic queue).
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from src.core import state as state_mod
from src.core.models import Context, Item, Result
from src.core.registry import enrichers, sources

logger = logging.getLogger(__name__)

LOOKBACK_HOURS = 48  # feed-scan window; dedup filters already-seen items on top


def gather(specs: list[dict], state: dict, since: datetime) -> list[Item]:
    """Return NEW items (state.is_new) published >= since, enriched per spec.

    A single source crashing is logged and skipped, never raised.
    """
    gathered: list[Item] = []
    for spec in specs:
        try:
            fetched = sources.get(spec["kind"])(spec, since, state)
        except Exception:
            logger.exception("❌ gather: source %s crashed", spec.get("key"))
            continue
        for item in fetched:
            if not state_mod.is_new(state, item) or item.published < since:
                continue
            for name in spec.get("enrich", []):
                item = enrichers.get(name)(item)
            gathered.append(item)
    return gathered


def run_source_task(
    ctx: Context, source_specs: list[dict], produce: Callable, subject: str
) -> Result:
    """Gather new items for `source_specs`, render them via `produce` -> markdown,
    and consume all gathered items."""
    since = datetime.now(UTC) - timedelta(hours=LOOKBACK_HOURS)
    items = ctx.gather(source_specs, since)
    ctx.log(f"{subject}: {len(items)} new item(s)")
    markdown = produce(ctx, items) if items else ""
    return Result(subject=subject, markdown=markdown, consumed=[it.id for it in items])
