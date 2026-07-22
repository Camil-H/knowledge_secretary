"""Deliverer: `site` — persists each task's daily output as JSON history
(committed to this repo, pruned to N days) and re-renders the last N days into
a single static HTML page. See CONTRACTS.md for signatures.

History writes and rendering may raise (run.py tolerates per-task failure).
`_upload_release_asset` degrades silently — a missing/failed GitHub release
shouldn't fail the run — logging any failure once here.
"""

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


# == Site =====================================================================


@deliverers.register("site")
def site(result: Result, cfg: dict) -> None:
    """Store today's result under history_dir keyed by task, prune, re-render."""
    conf = cfg["delivery"]["site"]
    task = result.meta.get("task", "")
    if not result.markdown and not result.artifacts:
        logger.info("site: nothing to add for task %s", task)
        return

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    entry = _load_entry(conf["history_dir"], today)

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
    _save_entry(conf["history_dir"], today, entry)
    _prune(conf["history_dir"], conf.get("history_days", 7))
    _render(conf)
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


def _render(conf: dict) -> None:
    history_dir = conf["history_dir"]
    out_dir = conf["out_dir"]
    history_days = conf.get("history_days", 7)

    entries = []
    for path in glob.glob(os.path.join(history_dir, "*.json")):
        with open(path) as f:
            entries.append(json.load(f))
    entries.sort(key=lambda e: e["date"], reverse=True)
    entries = entries[:history_days]

    days_html = "\n".join(_render_day(entry, is_latest=(i == 0)) for i, entry in enumerate(entries))
    page = string.Template(_PAGE).substitute(
        title=conf.get("title", ""),
        subtitle=conf.get("subtitle", ""),
        updated=datetime.now(UTC).isoformat(),
        days=days_html,
    )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(page)


def _render_day(entry: dict, *, is_latest: bool) -> str:
    tasks_html = "".join(
        _task_html(task, entry["tasks"][task]) for task in _LABELS if task in entry["tasks"]
    )
    if is_latest:
        return f'<section class="day today"><h2>{entry["date"]}</h2>{tasks_html}</section>'
    return f'<details class="day"><summary>{entry["date"]}</summary>{tasks_html}</details>'


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
    return f'<article class="task {task}"><h3>{label}</h3>{body}</article>'


# ----- podcast release upload -----


def _upload_release_asset(mp3_path: str, subject: str, topic: str, repo: str) -> str | None:
    """Create (or update) a dated GH release with mp3_path attached; return the
    asset's public download URL, or None on failure."""
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
