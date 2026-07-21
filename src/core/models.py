"""Frozen data contracts shared by every source, task, and deliverer.

DO NOT change field names/semantics without updating CONTRACTS.md — the source,
task, and deliverer modules code directly against these.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

# == Contracts ================================================================


@dataclass
class Item:
    """One unit of content from any source. `published` MUST be tz-aware UTC.

    `id` MUST be globally unique and source-prefixed for dedup, e.g.
    "rss:<link>", "pubmed:<pmid>", "biorxiv:<doi>", "x:<tweet_id>", "yt:<video_id>".
    """

    id: str
    source: str  # per-task source spec key, e.g. "pipeline", "yt_physionic"
    section: str  # grouping label for output, e.g. "Biology & Health"
    title: str
    url: str
    published: datetime  # tz-aware UTC
    text: str = ""  # full text / transcript / abstract / teaser (may be empty)
    meta: dict = field(default_factory=dict)  # channel, authors, lang, ...


@dataclass
class Result:
    """What a task returns to the dispatcher for delivery."""

    subject: str = ""
    markdown: str = ""
    artifacts: list[str] = field(default_factory=list)  # file paths, e.g. podcast mp3
    meta: dict = field(default_factory=dict)  # e.g. {"topic": ...} for the feed
    # Item ids to mark seen ONLY after this Result is delivered successfully, so a
    # failed send doesn't burn that run's content (it resurfaces next run).
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
    # gather(specs: list[dict], since: datetime) -> list[Item]
    #   Fetch NEW (deduped) items for the task's own source specs, published >= since.
    #   Does NOT mark items seen — the caller marks only what it actually consumes.
    gather: Callable
    # call(tier: str, system: str, user: str, *, max_tokens: int | None = None) -> str
    #   tier in {"summarize", "podcast"}. Tiered LLM call with free-model fallback.
    call: Callable
    log: Callable  # log(msg: str) -> None
