import glob
import json
import logging
import os
from datetime import UTC, datetime, timedelta

import pytest

from src.core.models import Result
from src.delivery import site


class _Resp:
    """Stand-in for a `subprocess.CompletedProcess`."""

    def __init__(self, returncode: int, stderr: str = "", stdout: str = ""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


class _RecordingRun:
    """Stub for `subprocess.run`: records every argv and replays canned responses in order."""

    def __init__(self, *responses: _Resp):
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs) -> _Resp:
        self.calls.append(argv)
        return self.responses.pop(0)


def _todays_tag() -> str:
    return site.RELEASE_TAG_PREFIX + datetime.now(UTC).strftime("%Y-%m-%d")


@pytest.fixture(autouse=True)
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(site, "HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setattr(site, "OUT_DIR", str(tmp_path / "public"))


def _index_html():
    with open(os.path.join(site.OUT_DIR, "index.html")) as f:
        return f.read()


def _rendered(result: Result) -> str:
    """Record a result then render, i.e. what a task run plus the publish step do."""
    site.site(result)
    site.render()
    return _index_html()


def _todays_tasks() -> dict:
    return site._load_entry(site.HISTORY_DIR, datetime.now(UTC).strftime("%Y-%m-%d"))["tasks"]


# ----- site: markdown tasks -----


def test_site_stores_newsletter_and_renders_today_expanded():
    html = _rendered(Result(subject="Digest", markdown="# Hello", meta={"task": "newsletter"}))
    assert "<h1>Hello</h1>" in html
    assert '<section class="day today">' in html
    assert "<details" not in html  # only one day so far


def test_site_second_task_upserts_same_day():
    site.site(Result(markdown="# News", meta={"task": "newsletter"}))
    html = _rendered(Result(markdown="# Vids", meta={"task": "youtube"}))
    assert '<article class="task newsletter">' in html
    assert '<article class="task youtube">' in html
    assert "<h1>News</h1>" in html and "<h1>Vids</h1>" in html


def test_site_does_not_render_at_record_time():
    # The page is rendered by the publish step, from the reconciled history: a task run that
    # rendered its own page would deploy one missing the other job's cards.
    site.site(Result(markdown="# News", meta={"task": "newsletter"}))
    assert "newsletter" in _todays_tasks()
    assert not os.path.exists(os.path.join(site.OUT_DIR, "index.html"))


def test_site_podcast_uploads_and_renders_audio(monkeypatch):
    monkeypatch.setattr(site, "_upload_release_asset", lambda *a, **k: "https://fake/ep.mp3")
    html = _rendered(
        Result(subject="Ep 1", artifacts=["ep.mp3"], meta={"task": "podcast", "topic": "PROTACs"})
    )
    assert '<audio controls src="https://fake/ep.mp3">' in html
    assert "PROTACs" in html


def test_site_podcast_degrades_to_none_url_when_upload_fails(monkeypatch):
    monkeypatch.setattr(site, "_upload_release_asset", lambda *a, **k: None)
    html = _rendered(
        Result(subject="Ep 1", artifacts=["ep.mp3"], meta={"task": "podcast", "topic": "PROTACs"})
    )
    assert _todays_tasks()["podcast"]["audio_url"] is None
    assert "(audio unavailable)" in html


def test_site_empty_result_is_noop():
    site.site(Result(meta={"task": "newsletter"}))
    assert not glob.glob(os.path.join(site.HISTORY_DIR, "*.json"))


def test_site_renders_notice_banner():
    html = _rendered(
        Result(markdown="# Hi", notices=["x_biotech: creds expired"], meta={"task": "newsletter"})
    )
    assert 'class="notice"' in html
    assert "x_biotech: creds expired" in html


def test_site_notices_only_result_still_records():
    # X auth failing with no other new content must still surface, not be dropped as "empty"
    html = _rendered(
        Result(markdown="", notices=["x_biotech: creds expired"], meta={"task": "newsletter"})
    )
    assert _todays_tasks()["newsletter"]["notices"] == ["x_biotech: creds expired"]
    assert "x_biotech: creds expired" in html


# ----- render -----


def _write_day(date: str, markdown: str) -> None:
    site._save_entry(
        site.HISTORY_DIR,
        date,
        {"date": date, "tasks": {"newsletter": {"kind": "markdown", "markdown": markdown}}},
    )


def test_render_orders_days_desc_newest_first():
    for date, subject in [("2026-07-19", "Old"), ("2026-07-20", "Mid"), ("2026-07-21", "New")]:
        _write_day(date, subject)
    site.render()
    html = _index_html()
    assert html.index('<section class="day today">') < html.index("<details")
    assert html.index("2026-07-21") < html.index("2026-07-20") < html.index("2026-07-19")


def test_render_includes_a_day_this_process_never_delivered():
    # Stands in for the other daily job's card, landing in history via the rebase reconcile:
    # render reads the history dir, so it can never publish a page that drops it.
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    site.site(Result(markdown="# Mine", meta={"task": "youtube"}))
    entry = site._load_entry(site.HISTORY_DIR, today)
    entry["tasks"]["podcast"] = {"kind": "podcast", "topic": "Theirs", "audio_url": None}
    site._save_entry(site.HISTORY_DIR, today, entry)

    site.render()

    html = _index_html()
    assert '<article class="task youtube">' in html
    assert '<article class="task podcast">' in html
    assert "Theirs" in html


def test_render_keeps_only_the_newest_history_days(monkeypatch):
    monkeypatch.setattr(site, "HISTORY_DAYS", 2)
    for date in ["2026-07-19", "2026-07-20", "2026-07-21"]:
        _write_day(date, f"Day {date}")
    site.render()
    html = _index_html()
    assert "2026-07-19" not in html
    assert "2026-07-20" in html and "2026-07-21" in html


# ----- render: what the log reports -----


def test_render_logs_the_newest_days_cards(caplog):
    _write_day("2026-07-26", "Older")
    site._save_entry(
        site.HISTORY_DIR,
        "2026-07-27",
        {
            "date": "2026-07-27",
            "tasks": {
                "podcast": {"kind": "podcast", "topic": "t", "audio_url": None},
                "newsletter": {"kind": "markdown", "markdown": "# N"},
            },
        },
    )

    with caplog.at_level(logging.INFO, logger="src.delivery.site"):
        site.render()

    # display order, not history key order, and only the newest day is named
    assert "rendered 2 day(s)" in caplog.text
    assert "2026-07-27: newsletter, podcast" in caplog.text
    assert "2026-07-26" not in caplog.text


def test_render_log_names_a_day_that_rendered_no_cards(caplog):
    site._save_entry(site.HISTORY_DIR, "2026-07-27", {"date": "2026-07-27", "tasks": {}})

    with caplog.at_level(logging.INFO, logger="src.delivery.site"):
        site.render()

    assert "2026-07-27: no cards" in caplog.text


def test_render_log_omits_the_day_note_without_history(caplog):
    with caplog.at_level(logging.INFO, logger="src.delivery.site"):
        site.render()

    assert "rendered 0 day(s)" in caplog.text
    assert "—" not in caplog.text


def test_day_cards_reports_exactly_what_render_day_emits():
    # the log is only trustworthy if it can't drift from the HTML: an unlabelled task key
    # renders no card, so it must not be reported as one either
    entry = {
        "date": "2026-07-27",
        "tasks": {
            "podcast": {"kind": "podcast", "topic": "t", "audio_url": None},
            "newsletter": {"kind": "markdown", "markdown": "# N"},
            "mystery": {"kind": "markdown", "markdown": "# ?"},
        },
    }
    cards = site._day_cards(entry)
    html_out = site._render_day(entry, is_latest=True)

    assert cards == ["newsletter", "podcast"]
    assert html_out.count("<article") == len(cards)
    assert "mystery" not in html_out


# ----- site: render-boundary XSS hardening -----


def test_task_html_neutralizes_hostile_markdown():
    payload = {
        "kind": "markdown",
        "markdown": (
            "<script>alert(1)</script>[x](javascript:alert(1))\n\n<img src=x onerror=alert(1)>"
        ),
    }
    html_out = site._task_html("newsletter", payload)
    assert "<script>" not in html_out
    assert "javascript:" not in html_out
    assert "onerror" not in html_out


def test_task_html_escapes_hostile_notice():
    payload = {"kind": "markdown", "markdown": "", "notices": ["<script>alert(1)</script>"]}
    html_out = site._task_html("newsletter", payload)
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_task_html_escapes_hostile_topic():
    payload = {"kind": "podcast", "topic": "<img src=x onerror=alert(1)>", "audio_url": None}
    html_out = site._task_html("podcast", payload)
    assert "<img" not in html_out
    assert "&lt;img" in html_out


@pytest.mark.parametrize(
    "url, is_rendered",
    [
        ("https://ok/ep.mp3", True),
        ("http://ok/ep.mp3", True),
        ("javascript:alert(1)", False),
        ("data:audio/mp3;base64,AAAA", False),
        (None, False),
    ],
)
def test_task_html_audio_url_scheme_is_validated(url, is_rendered):
    html_out = site._task_html("podcast", {"kind": "podcast", "topic": "", "audio_url": url})
    if is_rendered:
        assert f'src="{url}"' in html_out
    else:
        assert "<audio" not in html_out
        assert "(audio unavailable)" in html_out


# ----- body renderers -----


def test_task_html_resolves_a_newly_registered_kind():
    site._body_renderers.register("banner")(lambda p: f'<div class="banner">{p["text"]}</div>')
    html_out = site._task_html("newsletter", {"kind": "banner", "text": "hi"})
    assert '<div class="banner">hi</div>' in html_out


# ----- helpers -----


def test_prune_keeps_only_n_most_recent(tmp_path):
    history_dir = tmp_path / "prune"
    for date in ["2026-07-15", "2026-07-16", "2026-07-17", "2026-07-18"]:
        site._save_entry(str(history_dir), date, {"date": date, "tasks": {}})
    site._prune(str(history_dir), 2)
    remaining = sorted(os.path.basename(p) for p in glob.glob(str(history_dir / "*.json")))
    assert remaining == ["2026-07-17.json", "2026-07-18.json"]


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
    assert argv[:3] == ["gh", "release", "create"]
    assert argv[argv.index("--repo") + 1] == "org/repo"
    assert "--title=Subject" in argv
    assert "--notes=Topic" in argv
    # positionals (tag, file) come after a "--" so they can never be parsed as flags
    assert argv[-3:] == ["--", tag, "ep.mp3"]


def test_upload_release_asset_same_day_rerun_recovers_via_clobber_upload(monkeypatch):
    run = _RecordingRun(_Resp(1, stderr="release already exists"), _Resp(0))
    monkeypatch.setattr(site.subprocess, "run", run)

    url = site._upload_release_asset("ep.mp3", "S", "T", "org/repo")

    tag = _todays_tag()
    assert url == f"https://github.com/org/repo/releases/download/{tag}/ep.mp3"
    upload_argv = run.calls[1]
    assert upload_argv[:3] == ["gh", "release", "upload"]
    assert "--clobber" in upload_argv
    assert upload_argv[-3:] == ["--", tag, "ep.mp3"]


def test_upload_release_asset_neutralizes_flag_shaped_subject_and_path(monkeypatch):
    # A subject/topic (or mp3 path) that looks like a CLI flag must not be parsed as one.
    run = _RecordingRun(_Resp(0))
    monkeypatch.setattr(site.subprocess, "run", run)

    site._upload_release_asset("--rf", "--json", "-rf", "org/repo")

    argv = run.calls[0]
    assert "--title=--json" in argv
    assert "--notes=-rf" in argv
    tag = _todays_tag()
    assert argv[-3:] == ["--", tag, "--rf"]  # positional mp3 path, shielded by "--"


def test_upload_release_asset_subprocess_exception_returns_none(monkeypatch):
    def _raise(*a, **k):
        raise OSError("gh executable not found")

    monkeypatch.setattr(site.subprocess, "run", _raise)
    assert site._upload_release_asset("ep.mp3", "S", "T", "org/repo") is None


# ----- release pruning -----


def _tag(days_ago: int) -> str:
    return (
        site.RELEASE_TAG_PREFIX + (datetime.now(UTC).date() - timedelta(days=days_ago)).isoformat()
    )


def test_prune_old_releases_deletes_only_old_podcast_tags(monkeypatch):
    old, recent = _tag(30), _tag(2)
    listing = json.dumps([{"tagName": old}, {"tagName": recent}, {"tagName": "v1.0.0"}])
    calls = []

    def _run(argv, **kwargs):
        calls.append(argv)
        return _Resp(0, stdout=listing) if argv[:3] == ["gh", "release", "list"] else _Resp(0)

    monkeypatch.setattr(site.subprocess, "run", _run)
    site._prune_old_releases("org/repo", 7)

    deletes = [a for a in calls if a[:3] == ["gh", "release", "delete"]]
    assert len(deletes) == 1  # only the 30-day-old podcast release
    assert old in deletes[0] and "--cleanup-tag" in deletes[0]
    assert not any(recent in a or "v1.0.0" in a for a in deletes)  # recent + non-podcast untouched


def test_prune_old_releases_noop_without_repo(monkeypatch):
    def _fail(*a, **k):
        raise AssertionError("must not shell out without a repo")

    monkeypatch.setattr(site.subprocess, "run", _fail)
    site._prune_old_releases("", 7)  # no exception == pass


def test_prune_old_releases_degrades_on_list_failure(monkeypatch):
    calls = []

    def _run(argv, **kwargs):
        calls.append(argv)
        return _Resp(1)  # gh release list fails

    monkeypatch.setattr(site.subprocess, "run", _run)
    site._prune_old_releases("org/repo", 7)
    assert not any(a[:3] == ["gh", "release", "delete"] for a in calls)
