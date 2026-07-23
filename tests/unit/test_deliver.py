import glob
import os
from datetime import UTC, datetime

import pytest

from src.core.models import Result
from src.delivery import site


class _Resp:
    """Stand-in for a `subprocess.CompletedProcess`."""

    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


class _RecordingRun:
    """Stub for `subprocess.run`: records every argv and replays canned responses in order."""

    def __init__(self, *responses: _Resp):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs) -> _Resp:
        self.calls.append(argv)
        return self.responses.pop(0)


def _todays_tag() -> str:
    return "podcast-" + datetime.now(UTC).strftime("%Y-%m-%d")


@pytest.fixture(autouse=True)
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setattr(site, "OUT_DIR", str(tmp_path / "public"))


def _index_html():
    with open(os.path.join(site.OUT_DIR, "index.html")) as f:
        return f.read()


# ----- site: markdown tasks -----


def test_site_stores_newsletter_and_renders_today_expanded():
    site.site(Result(subject="Digest", markdown="# Hello", meta={"task": "newsletter"}))
    html = _index_html()
    assert "<h1>Hello</h1>" in html
    assert '<section class="day today">' in html
    assert "<details" not in html  # only one day so far


def test_site_second_task_upserts_same_day():
    site.site(Result(markdown="# News", meta={"task": "newsletter"}))
    site.site(Result(markdown="# Vids", meta={"task": "youtube"}))
    html = _index_html()
    assert '<article class="task newsletter">' in html
    assert '<article class="task youtube">' in html
    assert "<h1>News</h1>" in html and "<h1>Vids</h1>" in html


def test_site_podcast_uploads_and_renders_audio(monkeypatch):
    monkeypatch.setattr(site, "_upload_release_asset", lambda *a, **k: "https://fake/ep.mp3")
    site.site(
        Result(subject="Ep 1", artifacts=["ep.mp3"], meta={"task": "podcast", "topic": "PROTACs"})
    )
    html = _index_html()
    assert '<audio controls src="https://fake/ep.mp3">' in html
    assert "PROTACs" in html


def test_site_podcast_degrades_to_none_url_when_upload_fails(monkeypatch):
    monkeypatch.setattr(site, "_upload_release_asset", lambda *a, **k: None)
    site.site(
        Result(subject="Ep 1", artifacts=["ep.mp3"], meta={"task": "podcast", "topic": "PROTACs"})
    )
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    entry = site._load_entry(site.HISTORY_DIR, today)
    assert entry["tasks"]["podcast"]["audio_url"] is None
    html = _index_html()
    assert "(audio unavailable)" in html


def test_site_empty_result_is_noop():
    site.site(Result(meta={"task": "newsletter"}))
    assert not glob.glob(os.path.join(site.HISTORY_DIR, "*.json"))
    assert not os.path.exists(os.path.join(site.OUT_DIR, "index.html"))


def test_site_renders_notice_banner():
    site.site(
        Result(markdown="# Hi", notices=["x_biotech: creds expired"], meta={"task": "newsletter"})
    )
    html = _index_html()
    assert 'class="notice"' in html
    assert "x_biotech: creds expired" in html


def test_site_notices_only_result_still_records():
    # X auth failing with no other new content must still surface, not be dropped as "empty"
    site.site(
        Result(markdown="", notices=["x_biotech: creds expired"], meta={"task": "newsletter"})
    )
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    entry = site._load_entry(site.HISTORY_DIR, today)
    assert entry["tasks"]["newsletter"]["notices"] == ["x_biotech: creds expired"]
    assert "x_biotech: creds expired" in _index_html()


# ----- helpers -----


def test_prune_keeps_only_n_most_recent(tmp_path):
    history_dir = tmp_path / "prune"
    for date in ["2026-07-15", "2026-07-16", "2026-07-17", "2026-07-18"]:
        site._save_entry(str(history_dir), date, {"date": date, "tasks": {}})
    site._prune(str(history_dir), 2)
    remaining = sorted(os.path.basename(p) for p in glob.glob(str(history_dir / "*.json")))
    assert remaining == ["2026-07-17.json", "2026-07-18.json"]


def test_render_orders_days_desc_newest_first():
    for date, subject in [("2026-07-19", "Old"), ("2026-07-20", "Mid"), ("2026-07-21", "New")]:
        site._save_entry(
            site.HISTORY_DIR,
            date,
            {"date": date, "tasks": {"newsletter": {"kind": "markdown", "markdown": subject}}},
        )
    site._render()
    html = _index_html()
    assert html.index('<section class="day today">') < html.index("<details")
    assert html.index("2026-07-21") < html.index("2026-07-20") < html.index("2026-07-19")


def test_load_entry_missing_file_returns_empty_shape(tmp_path):
    assert site._load_entry(str(tmp_path), "2026-07-21") == {"date": "2026-07-21", "tasks": {}}


def test_task_html_renders_in_fixed_order_skipping_absent():
    entry_tasks = {
        "podcast": {"kind": "podcast", "topic": "t", "audio_url": None},
        "newsletter": {"kind": "markdown", "markdown": "# N"},
    }
    html = "".join(site._task_html(t, entry_tasks[t]) for t in site._LABELS if t in entry_tasks)
    assert html.index('class="task newsletter"') < html.index('class="task podcast"')
    assert "(audio unavailable)" in html


def test_upload_release_asset_returns_none_without_repo():
    assert site._upload_release_asset("ep.mp3", "S", "T", "") is None


def test_upload_release_asset_gh_error_degrades_to_none(monkeypatch):
    class _R:
        returncode = 1
        stderr = "boom"

    monkeypatch.setattr(site.subprocess, "run", lambda *a, **k: _R())
    assert site._upload_release_asset("ep.mp3", "S", "T", "org/repo") is None


def test_upload_release_asset_happy_path_returns_url_and_composes_create_argv(monkeypatch):
    run = _RecordingRun(_Resp(0))
    monkeypatch.setattr(site.subprocess, "run", run)

    url = site._upload_release_asset("ep.mp3", "Subject", "Topic", "org/repo")

    tag = _todays_tag()
    assert url == f"https://github.com/org/repo/releases/download/{tag}/ep.mp3"
    argv = run.calls[0]
    assert argv[:4] == ["gh", "release", "create", tag]
    assert "ep.mp3" in argv
    assert argv[argv.index("--repo") + 1] == "org/repo"
    assert argv[argv.index("--title") + 1] == "Subject"
    assert argv[argv.index("--notes") + 1] == "Topic"


def test_upload_release_asset_same_day_rerun_recovers_via_clobber_upload(monkeypatch):
    run = _RecordingRun(_Resp(1, stderr="release already exists"), _Resp(0))
    monkeypatch.setattr(site.subprocess, "run", run)

    url = site._upload_release_asset("ep.mp3", "S", "T", "org/repo")

    tag = _todays_tag()
    assert url == f"https://github.com/org/repo/releases/download/{tag}/ep.mp3"
    upload_argv = run.calls[1]
    assert upload_argv[:4] == ["gh", "release", "upload", tag]
    assert "--clobber" in upload_argv


def test_upload_release_asset_subprocess_exception_returns_none(monkeypatch):
    def _raise(*a, **k):
        raise OSError("gh executable not found")

    monkeypatch.setattr(site.subprocess, "run", _raise)
    assert site._upload_release_asset("ep.mp3", "S", "T", "org/repo") is None
