"""Deliverer `site`: append each task's daily output to JSON history (committed,
pruned to N days) and re-render the last N days into one static HTML page."""

import glob
import json
import logging
import os
import string
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import markdown

from src.core.models import Result
from src.core.registry import deliverers

logger = logging.getLogger(__name__)

_LABELS = {"newsletter": "Newsletter", "youtube": "YouTube", "podcast": "Podcast"}
_PAGE = (Path(__file__).parent / "template.html").read_text()

TITLE = "Knowledge Secretary"
SUBTITLE = "Daily newsletter, YouTube digest, and technical podcast"
HISTORY_DIR = "history"
HISTORY_DAYS = 7
OUT_DIR = "public"


# == Site =====================================================================


@deliverers.register("site")
def site(result: Result) -> None:
    """Store today's result under HISTORY_DIR keyed by task, prune, re-render."""
    task = result.meta.get("task", "")
    if not result.markdown and not result.artifacts:
        logger.info("site: nothing to add for task %s", task)
        return

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    entry = _load_entry(HISTORY_DIR, today)

    if result.artifacts:
        episode_repo = os.environ.get("GITHUB_REPOSITORY", "")
        audio_url = _upload_release_asset(
            result.artifacts[0], result.subject, result.meta.get("topic", ""), episode_repo
        )
        payload = {
            "kind": "podcast",
            "subject": result.subject,
            "topic": result.meta.get("topic", ""),
            "audio_url": audio_url,
        }
    else:
        payload = {"kind": "markdown", "subject": result.subject, "markdown": result.markdown}

    entry["tasks"][task] = payload
    _save_entry(HISTORY_DIR, today, entry)
    _prune(HISTORY_DIR, HISTORY_DAYS)
    _render()
    logger.info("✅ site: recorded task %s for %s", task, today)


# == Helper Functions =========================================================

# ----- history -----


def _load_entry(history_dir: str, date: str) -> dict:
    path = os.path.join(history_dir, f"{date}.json")
    if not os.path.exists(path):
        return {"date": date, "tasks": {}}
    with open(path) as f:
        return json.load(f)


def _save_entry(history_dir: str, date: str, entry: dict) -> None:
    os.makedirs(history_dir, exist_ok=True)
    path = os.path.join(history_dir, f"{date}.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2, sort_keys=True)


def _prune(history_dir: str, days: int) -> None:
    files = sorted(glob.glob(os.path.join(history_dir, "*.json")))
    for path in files[:-days] if days > 0 else files:
        os.remove(path)


# ----- rendering -----


def _render() -> None:
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
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(page)


def _render_day(entry: dict, *, is_latest: bool) -> str:
    tasks_html = "".join(
        _task_html(task, entry["tasks"][task]) for task in _LABELS if task in entry["tasks"]
    )
    date = entry["date"]
    heading = f'<time class="js-date" datetime="{date}">{date}</time>'
    if is_latest:
        return f'<section class="day today"><h2>{heading}</h2>{tasks_html}</section>'
    return f'<details class="day"><summary>{heading}</summary>{tasks_html}</details>'


def _task_html(task: str, payload: dict) -> str:
    label = _LABELS.get(task, task)
    if payload.get("kind") == "podcast":
        audio_url = payload.get("audio_url")
        audio_html = (
            f'<audio controls src="{audio_url}"></audio>'
            if audio_url
            else "<p>(audio unavailable)</p>"
        )
        body = f'<p class="topic">{payload.get("topic", "")}</p>{audio_html}'
    else:
        body = markdown.markdown(payload.get("markdown", ""), extensions=["extra"])
    return f'<article class="task {task}"><h3 class="task-label">{label}</h3>{body}</article>'


# ----- podcast release upload -----


def _upload_release_asset(mp3_path: str, subject: str, topic: str, repo: str) -> str | None:
    """Attach mp3 to a dated GH release; return its public download URL or None."""
    if not repo:
        logger.warning("⚠️ site: no episode_repo configured, skipping podcast upload")
        return None

    tag = "podcast-" + datetime.now(UTC).strftime("%Y-%m-%d")
    title = subject or topic or tag
    notes = topic or title
    try:
        create = subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                mp3_path,
                "--repo",
                repo,
                "--title",
                title,
                "--notes",
                notes,
            ],
            capture_output=True,
            text=True,
        )
        if create.returncode != 0:
            # most likely today's tag already exists (same-day rerun) -> replace asset.
            # NOTE: also catches genuine auth/repo errors, which then fail the upload below.
            upload = subprocess.run(
                ["gh", "release", "upload", tag, mp3_path, "--repo", repo, "--clobber"],
                capture_output=True,
                text=True,
            )
            if upload.returncode != 0:
                logger.warning(
                    "⚠️ site: gh release create+upload failed: %s / %s",
                    create.stderr.strip(),
                    upload.stderr.strip(),
                )
                return None
    except Exception as e:
        logger.warning("⚠️ site: gh release error: %s", e)
        return None

    return f"https://github.com/{repo}/releases/download/{tag}/{os.path.basename(mp3_path)}"
