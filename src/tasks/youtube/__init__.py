"""YouTube daily digest task.

Summarizes videos uploaded within a fixed ET clock window (yesterday 08:00 ET
through today 07:59:59 ET, inclusive) across the configured channels. Only
in-window videos are reported consumed, so an upload published after the window
end resurfaces in tomorrow's window instead of being burned.
"""

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.core import userdata
from src.core.models import Context, Item, Result
from src.core.registry import tasks
from src.tasks.youtube import adapters as _adapters  # noqa: F401  (registers adapter/enricher)

PROMPT = (Path(__file__).parent / "prompt.md").read_text()
TRANSCRIPT_CHAR_LIMIT = 12000
_NO_TRANSCRIPT = ["- (no transcript available)"]
SOURCES = userdata.load(Path(__file__).parent, [])


# == Task =====================================================================


@tasks.register("youtube")
def run(ctx: Context) -> Result:
    """Gather new uploads, keep those inside the ET window, summarize each, and
    render the grouped digest markdown."""
    task_cfg = ctx.cfg["tasks"]["youtube"]
    specs = SOURCES
    tz = ZoneInfo(ctx.cfg["timezone"])
    now_et = datetime.now(tz)
    start_utc, end_utc = _window_utc(now_et, task_cfg["window_et"], tz)

    items = ctx.gather(specs, start_utc)
    _audit(ctx, specs, items, start_utc, end_utc)
    in_window = [it for it in items if start_utc <= it.published <= end_utc]

    section_order = _section_order(specs)
    grouped: dict[str, list[tuple[Item, list[str]]]] = {}
    for item in in_window:
        grouped.setdefault(item.section, []).append((item, _summarize(ctx, item)))

    today = now_et.strftime("%Y-%m-%d")
    markdown = _render(today, section_order, grouped) if in_window else ""
    return Result(
        subject=f"YouTube Digest — {today}",
        markdown=markdown,
        consumed=[it.id for it in in_window],
    )


# == Helper Functions =========================================================


def _window_utc(now_et: datetime, window_et: dict, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """Compute [start, end] of the DST-safe ET clock window, in UTC."""
    start_h, start_m = (int(x) for x in window_et["start"].split(":"))
    end_h, end_m = (int(x) for x in window_et["end"].split(":"))
    yesterday = now_et.date() - timedelta(days=1)
    start_et = datetime.combine(yesterday, time(start_h, start_m, 0), tzinfo=tz)
    end_et = datetime.combine(now_et.date(), time(end_h, end_m, 59), tzinfo=tz)
    return start_et.astimezone(UTC), end_et.astimezone(UTC)


def _section_order(specs: list[dict]) -> list[str]:
    """Section names in the order they first appear across the task's specs."""
    order: list[str] = []
    for spec in specs:
        if spec["section"] not in order:
            order.append(spec["section"])
    return order


def _summarize(ctx: Context, item: Item) -> list[str]:
    """The model's bullet lines for one video, or a note if there's no transcript."""
    if not item.text:
        ctx.log(f"youtube: no transcript for '{item.title}' ({item.url}); skipping summarization")
        return _NO_TRANSCRIPT
    user = (
        f"Title: {item.title}\n"
        f"Channel: {item.meta.get('channel', '')}\n"
        f"Transcript:\n{item.text[:TRANSCRIPT_CHAR_LIMIT]}"
    )
    raw = ctx.call("summarize", system=PROMPT, user=user)
    return [line for line in raw.splitlines() if line.strip()]


def _audit(
    ctx: Context, specs: list[dict], items: list[Item], start_utc: datetime, end_utc: datetime
) -> None:
    """Log, per channel, the newest gathered upload and whether it fell in the window."""
    by_source: dict[str, list[Item]] = {}
    for it in items:
        by_source.setdefault(it.source, []).append(it)
    for spec in specs:
        channel_items = by_source.get(spec["key"], [])
        if not channel_items:
            ctx.log(f"youtube audit: {spec['key']}: no items gathered")
            continue
        newest = max(channel_items, key=lambda i: i.published)
        in_window = start_utc <= newest.published <= end_utc
        ctx.log(
            f"youtube audit: {newest.meta.get('channel', spec['key'])}: newest='{newest.title}' "
            f"published={newest.published.isoformat()} in_window={in_window}"
        )


def _render(
    today: str, section_order: list[str], grouped: dict[str, list[tuple[Item, list[str]]]]
) -> str:
    """Render the digest markdown, grouped by section in config order."""
    lines = [today]
    for section in section_order:
        entries = grouped.get(section) or []
        if not entries:
            continue
        lines.append(f"- {section}")
        for item, bullets in entries:
            lines.append(f"    - [{item.title}]({item.url}) -- {item.meta.get('channel', '')}")
            lines.extend(f"        {bullet}" for bullet in bullets)
    return "\n".join(lines)
