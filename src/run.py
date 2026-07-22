"""Entrypoint: python -m src.run [newsletter|youtube|podcast|all]."""

import logging
import sys

from src.core import llm
from src.core import state as state_mod
from src.core.models import Context
from src.core.registry import deliverers, tasks
from src.delivery import site as site
from src.tasks.runner import gather

logger = logging.getLogger("knowledge_secretary")


def build_context(state: dict) -> Context:
    return Context(
        state=state,
        gather=lambda specs, since: gather(specs, state, since),
        call=lambda system, user, max_tokens=None: llm.call(system, user, max_tokens=max_tokens),
        log=logger.info,
    )


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    which = argv[1] if len(argv) > 1 else "all"
    state = state_mod.load()

    names = list(tasks.all()) if which == "all" else [which]
    failures = []
    for name in names:
        try:
            _run_task(name, state)
        except Exception:  # one task failing must not sink the others
            failures.append(name)
            logger.exception("❌ task %s failed", name)

    state_mod.prune(state)
    state_mod.save(state)
    return 1 if failures else 0


def _run_task(name: str, state: dict) -> None:
    logger.info("🚀 running task %s", name)
    result = tasks.get(name)(build_context(state))
    result.meta.setdefault("task", name)
    deliverers.get("site")(result)
    # only now that delivery succeeded do we consider these items handled
    state_mod.mark_ids(state, result.consumed)
    logger.info("✅ task %s done (consumed=%d)", name, len(result.consumed))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
