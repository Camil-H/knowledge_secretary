"""Fetch driver (gather) + the load->gather->produce->consume shell shared by the
gather-based tasks (newsletter, youtube). The podcast uses neither."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from src.core import state as state_mod
from src.core.errors import AuthError
from src.core.models import Context, Item, Result
from src.core.registry import enrichers, sources

logger = logging.getLogger(__name__)

LOOKBACK_HOURS = 48  # feed-scan window; dedup filters already-seen items on top
_NOTICES_KEY = "_notices"  # transient: gather appends, run_source_task drains before state is saved


def gather(specs: list[dict], state: dict, since: datetime) -> list[Item]:
    """NEW items (is_new) published >= since, enriched per spec; crashing sources skipped."""
    gathered: list[Item] = []
    for spec in specs:
        try:
            fetched = sources.get(spec["kind"])(spec, since, state)
        except AuthError as e:
            logger.error("❌ gather: source %s auth failed — %s", spec.get("key"), e)
            notice = e.detail or "authentication failed"
            state.setdefault(_NOTICES_KEY, []).append(f"{spec.get('key')}: {notice}")
            continue
        except Exception:
            logger.exception("❌ gather: source %s crashed", spec.get("key"))
            continue
        kept = 0
        for item in fetched:
            if not state_mod.is_new(state, item) or item.published < since:
                continue
            for name in spec.get("enrich", []):
                item = enrichers.get(name)(item)
            gathered.append(item)
            kept += 1
        logger.info("gather: %s → %d new item(s)", spec.get("key"), kept)
    return gathered


def run_source_task(
    ctx: Context, source_specs: list[dict], produce: Callable, subject: str
) -> Result:
    """Gather new items, render via `produce` -> markdown, consume all gathered."""
    since = datetime.now(UTC) - timedelta(hours=LOOKBACK_HOURS)
    items = ctx.gather(source_specs, since)
    notices = ctx.state.pop(_NOTICES_KEY, [])
    ctx.logger.info(f"{subject}: {len(items)} new item(s)")
    markdown = produce(ctx, items) if items else ""
    return Result(
        subject=subject, markdown=markdown, notices=notices, consumed=[it.id for it in items]
    )
