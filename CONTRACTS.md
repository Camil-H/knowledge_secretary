# Frozen contracts — code against these EXACTLY

The core spine (`src/core/`) is already written and must not be modified by leaf
modules. Import from it; match these signatures precisely.

## Data types — `src/core/models.py`
```python
@dataclass
class Item:
    id: str            # UNIQUE, source-prefixed: "rss:<link>", "pubmed:<pmid>",
                       #   "biorxiv:<doi>", "x:<tweet_id>", "yt:<video_id>"
    source: str        # catalog key, e.g. "pipeline"
    section: str       # output grouping label, e.g. "Blogs"
    title: str
    url: str
    published: datetime  # MUST be tz-aware UTC (use datetime.now(timezone.utc) / astimezone(timezone.utc))
    text: str = ""
    meta: dict = {}

@dataclass
class Result:
    subject: str = ""; markdown: str = ""; artifacts: list[str] = []; meta: dict = {}
    consumed: list[str] = []   # item ids to mark seen ONLY after successful delivery

@dataclass
class Context:      # injected into tasks; reach network/LLM ONLY via these
    cfg: dict; state: dict
    gather(specs: list[dict], since: datetime) -> list[Item]   # per-task specs; does NOT mark
    call(tier: str, system: str, user: str, *, max_tokens: int | None = None) -> str
    log(msg: str) -> None
```
Tasks report the ids they actually used in `Result.consumed`; the dispatcher
(`run.py`) marks them seen only after every deliverer succeeds, so a failed send
never burns that run's content. Tasks MAY import `src.core.state` for dedup/KV.

## Registries — `src/core/registry.py`
```python
from src.core.registry import sources, enrichers, deliverers, tasks
```
Each is a `Registry`: `@sources.register("feed")` to add, `sources.get(name)` to fetch.

## State — `src/core/state.py`
`load()`, `is_new(state,item)->bool`, `mark(state,items)`, `mark_ids(state,ids)`,
`get_kv(state,k,default)`, `set_kv(state,k,v)`, `prune(state,days)`, `save(state)`.

## LLM — `src/core/llm.py`
`call(task, system, user, cfg, *, max_tokens=None) -> str` — already wired into `Context.call`.

## Signatures leaf modules MUST implement

### Source adapters (one `sources.py` per task, in `src/tasks/<task>/sources.py`)
```python
@sources.register("<kind>")
def adapter(spec: dict, since: datetime, state: dict) -> list[Item]:
    # spec is a per-task inline source dict incl. its own "key". Never raise on a
    # single source failing — log and return []. published must be tz-aware UTC.
```
Kinds by task:
- **newsletter/sources.py**: `feed` (plain RSS/Atom, spec["url"]), `pubmed` (spec["queries"]),
  `biorxiv` (spec["categories"]), `twitter` (agent-reach `twitter` backend, spec["handles"]; degrade to []).
- **youtube/sources.py**: `yt_channel` (resolve spec["handle"] → channel_id, cache in state KV
  "yt_channel:<handle>", read the uploads `videos.xml` feed).

Adapters are **thin mappers** over `src/fetchers/` — deterministic content
fetchers by source type (`rss`, `url`, `youtube`, `x`, `pubmed`, `biorxiv`), each
degrading gracefully (return []/None/"" + log on failure) and holding no state.
An adapter calls a fetcher and maps its raw output to `Item`s (stateful bits like
channel-id caching stay in the adapter). Adapters/enrichers register when the task
bucket is imported (each task `__init__` does `from . import sources`).

### Enrichers (in the task's `sources.py`)
```python
@enrichers.register("article_text")   # newsletter: trafilatura, set item.text
@enrichers.register("transcript")     # youtube: youtube-transcript-api, set item.text
def enrich(item: Item) -> Item:  ...  # never raise; return item unchanged on failure
```

### gather (in `src/core/gather.py`) — the one driver tasks call via Context.gather
```python
def gather(specs: list[dict], state: dict, since: datetime) -> list[Item]:
    # For each spec: dispatch to sources.get(spec["kind"]),
    # keep only is_new(state, item) AND item.published >= since,
    # run each enricher named in spec.get("enrich", []), return them.
    # Does NOT mark — the caller marks only what it consumes (Result.consumed).
```

### Deliverers (in `src/core/deliver.py`)
```python
@deliverers.register("site")
def site(result: Result, cfg: dict) -> None:
    # stores each task's daily output under delivery.site.history_dir (pruned to
    # history_days), renders the last N days to <out_dir>/index.html (newest day
    # expanded, older days behind <details>), uploads the podcast mp3 as a GH
    # Release asset and embeds an <audio> player. Reads the task name from
    # result.meta["task"] (set by run.py).
```

### Tasks (in `src/tasks/<name>/__init__.py`)
```python
from src.core.registry import tasks
@tasks.register("<name>")
def run(ctx: Context) -> Result: ...
# Load the bucket's own prompt with: (Path(__file__).parent / "prompt.md").read_text()
```
