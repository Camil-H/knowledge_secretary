"""Entrypoint: python -m src.run [newsletter|youtube|podcast|all]

Discovers registered tasks, builds a Context with injected helpers, runs the
requested task(s), routes each Result to the deliverers named in that task's
config, marks consumed items seen ONLY after successful delivery, then prunes +
saves dedup state.
"""

import logging
import sys

import src.tasks  # noqa: F401  (registers task buckets)
from src.core import config as config_mod
from src.core import gather as gather_mod
from src.core import llm
from src.core import state as state_mod
from src.core.models import Context
from src.core.registry import deliverers, tasks
from src.delivery import site as _site  # noqa: F401  (registers the site deliverer)

logger = logging.getLogger("knowledge_secretary")


def build_context(cfg: dict, state: dict) -> Context:
    return Context(
        cfg=cfg,
        state=state,
        gather=lambda specs, since: gather_mod.gather(specs, state, since),
        call=lambda tier, system, user, max_tokens=None: llm.call(
            tier, system, user, cfg, max_tokens=max_tokens
        ),
        log=logger.info,
    )


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    which = argv[1] if len(argv) > 1 else "all"
    cfg = config_mod.load()
    state = state_mod.load()

    names = list(tasks.all()) if which == "all" else [which]
    failures = []
    for name in names:
        try:
            _run_task(name, cfg, state)
        except Exception:  # one task failing must not sink the others
            failures.append(name)
            logger.exception("❌ task %s failed", name)

    state_mod.prune(state)
    state_mod.save(state)
    return 1 if failures else 0


def _run_task(name: str, cfg: dict, state: dict) -> None:
    logger.info("🚀 running task %s", name)
    result = tasks.get(name)(build_context(cfg, state))
    result.meta.setdefault("task", name)  # deliverers (e.g. site) key output by task
    for d in cfg["tasks"][name].get("deliver", []):
        deliverers.get(d)(result, cfg)
    # only now that delivery succeeded do we consider these items handled
    state_mod.mark_ids(state, result.consumed)
    logger.info(
        "✅ task %s done (delivered=%s, consumed=%d)",
        name,
        cfg["tasks"][name].get("deliver", []),
        len(result.consumed),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
