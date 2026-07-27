"""Deliverer `site`: append each task's daily output to JSON history (committed,
pruned to N days). Rendering the last N days into one static HTML page is a separate
entry point (`python -m src.delivery.site`), run at publish time — see `render`."""

import glob
import html
import json
import logging
import os
import string
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import markdown
import nh3

from src.core.models import Result
from src.core.registry import Registry, deliverers

logger = logging.getLogger(__name__)

# A day's history file: {"date": str, "tasks": {task: Payload}}. Payload is a
# per-task record whose keys vary by "kind" (markdown vs podcast).
type HistoryEntry = dict[str, Any]
type Payload = dict[str, Any]
type BodyRenderer = Callable[[Payload], str]

_LABELS = {"newsletter": "Newsletter", "youtube": "YouTube", "podcast": "Podcast"}
_PAGE = (Path(__file__).parent / "template.html").read_text()
# Rendered markdown is public-facing: keep formatting, drop script/handlers and any
# non-web href scheme. audio src= is validated separately when read back from history.
_MARKDOWN_URL_SCHEMES = {"http", "https", "mailto"}
_AUDIO_URL_SCHEMES = ("http", "https")

TITLE = os.environ.get("SITE_TITLE", "Knowledge Secretary")
SUBTITLE = os.environ.get(
    "SITE_SUBTITLE", "Daily newsletter, YouTube digest, and technical podcast"
)
HISTORY_DIR = "history"
HISTORY_DAYS = 7
OUT_DIR = "public"
RELEASE_TAG_PREFIX = "podcast-"


# == Site =====================================================================


@deliverers.register("site")
def site(result: Result) -> None:
    """Store today's result under HISTORY_DIR keyed by task, then prune.

    Recording only — the page is rendered later, by `render`, from the committed history."""
    task = result.meta.get("task", "")
    if not result.markdown and not result.artifacts and not result.notices:
        logger.info("site: nothing to add for task %s", task)
        return

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    entry = _load_entry(HISTORY_DIR, today)

    payload: Payload
    if result.artifacts:
        episode_repo = os.environ.get("GITHUB_REPOSITORY", "")
        audio_url = _upload_release_asset(
            result.artifacts[0], result.subject, result.meta.get("topic", ""), episode_repo
        )
        _prune_old_releases(episode_repo, HISTORY_DAYS)
        payload = {
            "kind": "podcast",
            "subject": result.subject,
            "topic": result.meta.get("topic", ""),
            "audio_url": audio_url,
        }
    else:
        payload = {"kind": "markdown", "subject": result.subject, "markdown": result.markdown}

    if result.notices:
        payload["notices"] = result.notices
    entry["tasks"][task] = payload
    _save_entry(HISTORY_DIR, today, entry)
    _prune(HISTORY_DIR, HISTORY_DAYS)
    logger.info("✅ site: recorded task %s for %s", task, today)


# == Render ===================================================================


def render() -> None:
    """Render the newest HISTORY_DAYS days of history into OUT_DIR/index.html.

    Reads the history dir rather than the Result just delivered, so the page is a pure
    function of what is committed. That is what lets the two daily jobs publish in any
    order: whichever renders last picks up the other's cards instead of dropping them.
    """
    entries = []
    for path in glob.glob(os.path.join(HISTORY_DIR, "*.json")):
        with open(path) as f:
            entries.append(json.load(f))
    entries.sort(key=lambda e: e["date"], reverse=True)
    entries = entries[:HISTORY_DAYS]

    days_html = "\n".join(_render_day(entry, is_latest=(i == 0)) for i, entry in enumerate(entries))
    page = string.Template(_PAGE).substitute(
        title=TITLE, subtitle=SUBTITLE, updated=datetime.now(UTC).isoformat(), days=days_html
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(page)
    # name the newest day's cards: the deployed page is what the run is judged on, and a
    # card missing here is the symptom worth seeing without diffing the published HTML
    logger.info(
        "✅ site: rendered %d day(s) to %s%s", len(entries), out_path, _latest_cards_note(entries)
    )


# == Helper Functions =========================================================

# ----- history -----


def _load_entry(history_dir: str, date: str) -> HistoryEntry:
    path = os.path.join(history_dir, f"{date}.json")
    if not os.path.exists(path):
        return {"date": date, "tasks": {}}
    with open(path) as f:
        return json.load(f)


def _save_entry(history_dir: str, date: str, entry: HistoryEntry) -> None:
    os.makedirs(history_dir, exist_ok=True)
    path = os.path.join(history_dir, f"{date}.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, sort_keys=True)


def _prune(history_dir: str, days: int) -> None:
    files = sorted(glob.glob(os.path.join(history_dir, "*.json")))
    for path in files[:-days] if days > 0 else files:
        os.remove(path)


# ----- rendering -----


def _day_cards(entry: HistoryEntry) -> list[str]:
    """The tasks this day renders a card for, in display order. Single source of truth for
    both the HTML and the log line, so what is reported can't drift from what is rendered."""
    return [task for task in _LABELS if task in entry.get("tasks", {})]


def _latest_cards_note(entries: list[HistoryEntry]) -> str:
    """' — <date>: <cards>' for the newest rendered day; empty when there is no history."""
    if not entries:
        return ""
    latest = entries[0]
    return f" — {latest['date']}: {', '.join(_day_cards(latest)) or 'no cards'}"


def _render_day(entry: HistoryEntry, *, is_latest: bool) -> str:
    tasks_html = "".join(_task_html(task, entry["tasks"][task]) for task in _day_cards(entry))
    date = html.escape(entry["date"])
    heading = f'<time class="js-date" datetime="{date}">{date}</time>'
    if is_latest:
        return f'<section class="day today"><h2>{heading}</h2>{tasks_html}</section>'
    return f'<details class="day"><summary>{heading}</summary>{tasks_html}</details>'


def _task_html(task: str, payload: Payload) -> str:
    label = html.escape(_LABELS.get(task, task))
    body = _body_renderers.get(payload["kind"])(payload)
    notices = "".join(
        f'<p class="notice">⚠️ {html.escape(n)}</p>' for n in payload.get("notices", [])
    )
    return (
        f'<article class="task {task}"><h3 class="task-label">{label}</h3>{notices}{body}</article>'
    )


# ----- body renderers (keyed by payload kind) -----

_body_renderers: Registry[BodyRenderer] = Registry("body renderer")


@_body_renderers.register("markdown")
def _markdown_body(payload: Payload) -> str:
    rendered = markdown.markdown(payload.get("markdown", ""), extensions=["extra"])
    return nh3.clean(rendered, url_schemes=_MARKDOWN_URL_SCHEMES)


@_body_renderers.register("podcast")
def _podcast_body(payload: Payload) -> str:
    audio_url = _safe_audio_url(payload.get("audio_url"))
    audio_html = (
        f'<audio controls src="{html.escape(audio_url)}"></audio>'
        if audio_url
        else "<p>(audio unavailable)</p>"
    )
    return f'<p class="topic">{html.escape(payload.get("topic", ""))}</p>{audio_html}'


def _safe_audio_url(url: str | None) -> str | None:
    """Guard the audio src= sink: accept only http(s) URLs read back from history JSON,
    so a javascript:/data: value can never reach the rendered attribute."""
    if url and urlsplit(url).scheme.lower() in _AUDIO_URL_SCHEMES:
        return url
    return None


# ----- podcast release upload -----


def _upload_release_asset(mp3_path: str, subject: str, topic: str, repo: str) -> str | None:
    """Attach mp3 to a dated GH release; return its public download URL or None."""
    if not repo:
        logger.warning("⚠️ site: no episode_repo configured, skipping podcast upload")
        return None

    tag = RELEASE_TAG_PREFIX + datetime.now(UTC).strftime("%Y-%m-%d")
    title = subject or topic or tag
    notes = topic or title
    try:
        # "--" ends flag parsing so tag/mp3_path (which may start with "-") aren't read as
        # flags; --title=/--notes= bind a "-"-leading value to its own flag.
        create = subprocess.run(
            [
                "gh",
                "release",
                "create",
                "--repo",
                repo,
                f"--title={title}",
                f"--notes={notes}",
                "--",
                tag,
                mp3_path,
            ],
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            # most likely today's tag already exists (same-day rerun) -> replace asset.
            # NOTE: also catches genuine auth/repo errors, which then fail the upload below.
            upload = subprocess.run(
                ["gh", "release", "upload", "--repo", repo, "--clobber", "--", tag, mp3_path],
                capture_output=True,
                text=True,
            )
            if upload.returncode != 0:
                logger.warning(
                    "⚠️ site: gh release create+upload failed: create exit=%s upload exit=%s",
                    create.returncode,
                    upload.returncode,
                )
                return None
    except Exception as e:
        logger.warning("⚠️ site: gh release error: %s", type(e).__name__)
        return None

    return f"https://github.com/{repo}/releases/download/{tag}/{os.path.basename(mp3_path)}"


def _prune_old_releases(repo: str, keep_days: int) -> None:
    """Delete podcast releases + tags older than keep_days so GH releases track the site's
    HISTORY_DAYS window — the audio is only linked while its day is still displayed."""
    if not repo:
        return
    cutoff = (datetime.now(UTC).date() - timedelta(days=keep_days)).isoformat()
    try:
        listed = subprocess.run(
            ["gh", "release", "list", "--repo", repo, "--limit", "1000", "--json", "tagName"],
            capture_output=True,
            text=True,
        )
    except OSError as e:
        logger.warning("⚠️ site: gh release list failed: %s", type(e).__name__)
        return
    if listed.returncode != 0:
        logger.warning("⚠️ site: gh release list failed: exit=%s", listed.returncode)
        return
    try:
        tags = [r.get("tagName", "") for r in json.loads(listed.stdout)]
    except ValueError:
        return
    for tag in tags:
        if tag.startswith(RELEASE_TAG_PREFIX) and tag[len(RELEASE_TAG_PREFIX) :] < cutoff:
            subprocess.run(
                ["gh", "release", "delete", "--repo", repo, "--yes", "--cleanup-tag", "--", tag],
                capture_output=True,
                text=True,
            )


# == Entry point ==============================================================


def main() -> int:
    """Render the page from the committed history. Invoked from the publish composite
    action once the state commit is reconciled and pushed, so the deployed page reflects
    what landed on the branch — not just the tasks this job happened to run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
