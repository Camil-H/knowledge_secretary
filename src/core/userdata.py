"""Load a task's source list / topic list from its committed `sources.yaml`.

The newsletter/youtube `sources.yaml` is a list of source-spec dicts; the
podcast's is a list of topic strings. Returns `default` if the file is missing
or empty.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_FILENAME = "sources.yaml"


def load(task_dir: Path, default: list | None = None):
    """Return the parsed list from <task_dir>/sources.yaml, or `default`."""
    path = task_dir / _FILENAME
    if not path.exists():
        logger.warning("⚠️ userdata: %s has no %s", task_dir.name, _FILENAME)
        return default
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else default
