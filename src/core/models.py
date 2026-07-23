"""Frozen data contracts shared by every source, task, and deliverer."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Item:
    """One unit of content. `published` is tz-aware UTC; `id` is globally unique and
    source-prefixed for dedup (e.g. "rss:<link>", "pubmed:<pmid>", "yt:<video_id>")."""

    id: str
    source: str
    section: str
    title: str
    url: str
    published: datetime
    text: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Result:
    subject: str = ""
    markdown: str = ""
    artifacts: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    # user-facing degradation notices rendered on the page (e.g. a source's creds expired)
    notices: list[str] = field(default_factory=list)
    # marked seen only after successful delivery, so a failed send doesn't burn the run
    consumed: list[str] = field(default_factory=list)


@dataclass
class Context:
    """Injected into every task's run(ctx): tasks reach LLM/gather/log only through
    these helpers, which keeps buckets self-contained and fake-testable."""

    state: dict
    gather: Callable
    call: Callable
    logger: logging.Logger
