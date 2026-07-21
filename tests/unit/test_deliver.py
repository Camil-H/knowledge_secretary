import glob
import os

from src.core import deliver
from src.core.models import Result

# ----- test doubles -----


def _cfg(tmp_path, *, history_days=7):
    return {
        "delivery": {
            "site": {
                "history_dir": str(tmp_path / "history"),
                "history_days": history_days,
                "out_dir": str(tmp_path / "public"),
                "title": "Knowledge Secretary",
                "subtitle": "Daily digest",
                "episode_repo": "org/repo",
            }
        }
    }


def _index_html(tmp_path) -> str:
    with open(tmp_path / "public" / "index.html") as f:
        return f.read()


# ----- site: markdown tasks -----


def test_site_stores_newsletter_and_renders_today_expanded(tmp_path):
    cfg = _cfg(tmp_path)
    result = Result(subject="Digest", markdown="# Hello", meta={"task": "newsletter"})

    deliver.site(result, cfg)

    html = _index_html(tmp_path)
    assert "<h1>Hello</h1>" in html
    assert '<section class="day today">' in html
    assert "<details" not in html  # only one day so far, no older days to collapse


def test_site_second_task_upserts_same_day(tmp_path):
    cfg = _cfg(tmp_path)
    deliver.site(Result(markdown="# News", meta={"task": "newsletter"}), cfg)
    deliver.site(Result(markdown="# Vids", meta={"task": "youtube"}), cfg)

    html = _index_html(tmp_path)
    assert '<article class="task newsletter">' in html
    assert '<article class="task youtube">' in html
    assert "<h1>News</h1>" in html
    assert "<h1>Vids</h1>" in html


def test_site_podcast_uploads_and_renders_audio(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(deliver, "_upload_release_asset", lambda *a, **k: "https://fake/ep.mp3")

    result = Result(
        subject="Ep 1", artifacts=["ep.mp3"], meta={"task": "podcast", "topic": "PROTACs"}
    )
    deliver.site(result, cfg)

    html = _index_html(tmp_path)
    assert '<audio controls src="https://fake/ep.mp3">' in html
    assert "PROTACs" in html


def test_site_empty_result_is_noop(tmp_path):
    cfg = _cfg(tmp_path)
    deliver.site(Result(meta={"task": "newsletter"}), cfg)

    assert not glob.glob(os.path.join(cfg["delivery"]["site"]["history_dir"], "*.json"))
    assert not os.path.exists(os.path.join(cfg["delivery"]["site"]["out_dir"], "index.html"))


# ----- _prune -----


def test_prune_keeps_only_n_most_recent(tmp_path):
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    for date in ["2026-07-15", "2026-07-16", "2026-07-17", "2026-07-18"]:
        deliver._save_entry(str(history_dir), date, {"date": date, "tasks": {}})

    deliver._prune(str(history_dir), 2)

    remaining = sorted(os.path.basename(p) for p in glob.glob(str(history_dir / "*.json")))
    assert remaining == ["2026-07-17.json", "2026-07-18.json"]


# ----- _render ordering -----


def test_render_orders_days_desc_newest_first(tmp_path):
    history_dir = tmp_path / "history"
    out_dir = tmp_path / "public"
    for date, subject in [("2026-07-19", "Old"), ("2026-07-20", "Mid"), ("2026-07-21", "New")]:
        deliver._save_entry(
            str(history_dir),
            date,
            {"date": date, "tasks": {"newsletter": {"kind": "markdown", "markdown": subject}}},
        )

    conf = {
        "history_dir": str(history_dir),
        "out_dir": str(out_dir),
        "history_days": 7,
        "title": "T",
        "subtitle": "S",
    }
    deliver._render(conf)

    with open(out_dir / "index.html") as f:
        html = f.read()

    today_idx = html.index('<section class="day today">')
    first_details_idx = html.index("<details")
    assert today_idx < first_details_idx
    assert html.index("2026-07-21") < html.index("2026-07-20") < html.index("2026-07-19")


# ----- unit-level helpers -----


def test_load_entry_missing_file_returns_empty_shape(tmp_path):
    entry = deliver._load_entry(str(tmp_path / "history"), "2026-07-21")
    assert entry == {"date": "2026-07-21", "tasks": {}}


def test_task_html_renders_in_fixed_order_skipping_absent():
    entry_tasks = {
        "podcast": {"kind": "podcast", "topic": "t", "audio_url": None},
        "newsletter": {"kind": "markdown", "markdown": "# N"},
    }
    html = "".join(
        deliver._task_html(t, entry_tasks[t]) for t in deliver._LABELS if t in entry_tasks
    )
    assert html.index('class="task newsletter"') < html.index('class="task podcast"')
    assert "(audio unavailable)" in html


def test_upload_release_asset_returns_none_without_repo():
    assert deliver._upload_release_asset("ep.mp3", "S", "T", "") is None


def test_upload_release_asset_gh_error_degrades_to_none(monkeypatch):
    class _R:
        returncode = 1
        stderr = "boom"

    monkeypatch.setattr(deliver.subprocess, "run", lambda *a, **k: _R())
    assert deliver._upload_release_asset("ep.mp3", "S", "T", "org/repo") is None
