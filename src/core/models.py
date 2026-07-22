"""Frozen data contracts shared by every source, task, and deliverer.

The source, task, and deliverer modules code directly against these — change
field names/semantics carefully.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Item:
    """One unit of content from any source. `published` MUST be tz-aware UTC.

    `id` MUST be globally unique and source-prefixed for dedup, e.g.
    "rss:<link>", "pubmed:<pmid>", "biorxiv:<doi>", "x:<tweet_id>", "yt:<video_id>".
    """

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
    # marked seen only after successful delivery, so a failed send doesn't burn the run
    consumed: list[str] = field(default_factory=list)


@dataclass
class Context:
    """Injected into every task's run(ctx).

    Tasks reach network/LLM work ONLY through these injected helpers (never
    importing `sources`/`llm` directly), which keeps buckets self-contained and
    unit-testable with fakes. Tasks MAY import `src.core.state` for pure dedup/KV
    dict ops (mark/get_kv/set_kv) — that is data manipulation, not I/O.
    """

    cfg: dict
    state: dict
    gather: Callable
    call: Callable
    log: Callable
