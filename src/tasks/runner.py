"""Fetch driver (gather) + the load->gather->produce->consume shell shared by the
gather-based tasks (newsletter, youtube). The podcast uses neither."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from src import config
from src.core import state as state_mod
from src.core.errors import AuthError
from src.core.models import Context, Item, Result, SourceSpec, State
from src.core.registry import enrichers, sources

logger = logging.getLogger(__name__)

_NOTICES_KEY = "_notices"  # transient: gather appends, run_source_task drains before state is saved


def gather(specs: list[SourceSpec], state: State, since: datetime) -> list[Item]:
    """NEW items (is_new) published >= since, enriched per spec; crashing sources skipped.

    Fetches run concurrently on a bounded pool, then enrichment does too — both are per-item
    network waits. Filtering and dedup stay on this thread, keeping state single-threaded, and
    every phase is collected in submission order so the output follows spec order."""
    pending: list[tuple[Item, list[str]]] = []

    with ThreadPoolExecutor(max_workers=_worker_count(len(specs))) as pool:
        futures = [pool.submit(_fetch_source_items, spec, since) for spec in specs]

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
        kept, seen, stale = 0, 0, 0
        for item in fetched:
            if not state_mod.is_new(state, item):
                seen += 1
                continue
            if item.published < since:
                stale += 1
                continue
            pending.append((item, spec.get("enrich", [])))
            kept += 1
        # the breakdown is what separates a source that returned nothing (fetched=0, i.e.
        # look at the fetcher) from one whose items were all already published or too old
        logger.info(
            "gather: %s → %d new of %d fetched (%d seen, %d outside window)",
            spec.get("key"),
            kept,
            len(fetched),
            seen,
            stale,
        )
    return _enrich_concurrently(pending)


def run_source_task(
    ctx: Context,
    source_specs: list[SourceSpec],
    produce: Callable[[Context, list[Item]], str],
    subject: str,
) -> Result:
    """Gather new items, render via `produce` -> markdown, consume all gathered."""
    since = datetime.now(UTC) - timedelta(hours=config.LOOKBACK_HOURS)
    items = ctx.gather(source_specs, since)
    notices = ctx.state.pop(_NOTICES_KEY, [])
    ctx.logger.info(f"{subject}: {len(items)} new item(s)")
    markdown = produce(ctx, items) if items else ""
    return Result(
        subject=subject, markdown=markdown, notices=notices, consumed=[it.id for it in items]
    )


# == Helper Functions =========================================================


def _fetch_source_items(spec: SourceSpec, since: datetime) -> list[Item]:
    return sources.get(spec["kind"])(spec, since)


def _enrich_concurrently(pending: list[tuple[Item, list[str]]]) -> list[Item]:
    """Each item enriched by its own spec's enrichers, returned in the order handed in."""
    if not pending:
        return []
    with ThreadPoolExecutor(max_workers=_worker_count(len(pending))) as pool:
        futures = [pool.submit(_enrich_item, item, names) for item, names in pending]
    return [future.result() for future in futures]


def _enrich_item(item: Item, names: list[str]) -> Item:
    """A failing enricher is tolerated: the item stays in the digest with whatever text it
    already had, which is what every enricher degrades to on its own bad days anyway."""
    for name in names:
        try:
            item = enrichers.get(name)(item)
        except Exception:
            logger.exception("❌ gather: enricher %s failed on %s", name, item.id)
    return item


def _worker_count(jobs: int) -> int:
    return min(config.MAX_FETCH_WORKERS, jobs) or 1
