"""Generic fetch + dedup + enrich driver shared by every task.

Task source modules (`src/tasks/<name>/sources.py`) register their adapters
(by `kind`) and enrichers into the core registries; `gather` is the single
driver each task calls via `Context.gather`. It never marks items seen — the
caller marks only what it actually consumes (Result.consumed).
"""

import logging
from datetime import datetime

from . import state as state_mod
from .models import Item
from .registry import enrichers, sources

logger = logging.getLogger(__name__)


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
