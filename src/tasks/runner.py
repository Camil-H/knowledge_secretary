"""Fetch driver (gather) + the load->gather->produce->consume shell shared by the
gather-based tasks (newsletter, youtube). The podcast uses neither."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from src.core import state as state_mod
from src.core.errors import AuthError
from src.core.models import Context, Item, Result, SourceSpec, State
from src.core.registry import enrichers, sources

logger = logging.getLogger(__name__)

LOOKBACK_HOURS = 48  # feed-scan window; dedup filters already-seen items on top
_NOTICES_KEY = "_notices"  # transient: gather appends, run_source_task drains before state is saved
_MAX_FETCH_WORKERS = 8


def gather(specs: list[SourceSpec], state: State, since: datetime) -> list[Item]:
    """NEW items (is_new) published >= since, enriched per spec; crashing sources skipped.

    Fetches run concurrently on a bounded pool; filtering, dedup and enrichment then run in
    spec order on this thread, keeping state single-threaded and output order deterministic."""
    gathered: list[Item] = []

    def _fetch(spec: SourceSpec) -> list[Item]:
        return sources.get(spec["kind"])(spec, since, state)

    workers = min(_MAX_FETCH_WORKERS, len(specs)) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch, spec) for spec in specs]

    for spec, future in zip(specs, futures, strict=True):
        try:
            fetched = future.result()
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
    ctx: Context,
    source_specs: list[SourceSpec],
    produce: Callable[[Context, list[Item]], str],
    subject: str,
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
