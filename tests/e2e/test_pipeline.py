"""Happy-path e2e tests: drive the real `src.run.main` entrypoint end to end
(registry -> gather -> produce -> deliver -> state) with every external
boundary faked via monkeypatch. See tests/e2e/conftest.py for the sandbox and
fake factories."""

import pytest

import src.run as run
from src.core import llm
from src.delivery import site
from src.tasks.newsletter import task as newsletter_task
from tests.e2e.conftest import (
    _install_all_llm,
    _install_newsletter_fakes,
    _install_podcast_fakes,
    _install_youtube_fakes,
    _raise,
    _read_history,
    _read_index,
    _read_state,
    _today,
)

# ----- newsletter -----


def test_newsletter_e2e_writes_history_renders_html_and_consumes(monkeypatch):
    seen: dict = {}
    _install_newsletter_fakes(monkeypatch)
    real_call = llm.call

    def _recording_call(system, user, max_tokens=None):
        seen["system"] = system
        seen["n"] = seen.get("n", 0) + 1
        return real_call(system, user, max_tokens=max_tokens)

    monkeypatch.setattr(llm, "call", _recording_call)

    code = run.main(["prog", "newsletter"])

    assert code == 0
    payload = _read_history(_today())["tasks"]["newsletter"]
    assert payload["kind"] == "markdown"
    assert "Knowledge Secretary" in payload["subject"]
    assert payload["markdown"] == "# Daily News\n\nBody"

    html = _read_index()
    assert "<h1>Daily News</h1>" in html
    assert '<article class="task newsletter">' in html
    assert '<section class="day today">' in html

    state = _read_state()
    assert state["ids"].get("rss:e1") == _today()

    # one editor pass over all gathered items, not one call per item
    assert seen["n"] == 1
    assert seen["system"] == newsletter_task.EDITOR_PROMPT


# ----- youtube -----


def test_youtube_e2e_summarizes_and_renders_grouped(monkeypatch):
    _install_youtube_fakes(monkeypatch)

    code = run.main(["prog", "youtube"])

    assert code == 0
    payload = _read_history(_today())["tasks"]["youtube"]
    assert payload["kind"] == "markdown"
    assert "[Vid A](http://yt/vid123)" in payload["markdown"]
    assert "ChanName" in payload["markdown"]
    assert "- key point" in payload["markdown"]

    html = _read_index()
    assert '<article class="task youtube">' in html
    assert "key point" in html

    state = _read_state()
    assert state["ids"].get("yt:vid123") == _today()


# ----- podcast -----


def test_podcast_e2e_uploads_release_asset_and_embeds_audio(monkeypatch, tmp_path):
    calls = _install_podcast_fakes(monkeypatch, tmp_path, topic="My Topic")

    code = run.main(["prog", "podcast"])

    assert code == 0
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:3] == ["gh", "release", "create"]
    assert f"{site.RELEASE_TAG_PREFIX}{_today()}" in argv
    assert str(tmp_path / "ep.mp3") in argv
    assert "--repo" in argv and "org/repo" in argv

    tag = f"{site.RELEASE_TAG_PREFIX}{_today()}"
    audio_url = f"https://github.com/org/repo/releases/download/{tag}/ep.mp3"
    payload = _read_history(_today())["tasks"]["podcast"]
    assert payload["kind"] == "podcast"
    assert payload["topic"] == "My Topic"
    assert payload["audio_url"] == audio_url

    html = _read_index()
    assert f'<audio controls src="{audio_url}"></audio>' in html
    assert "My Topic" in html

    state = _read_state()
    assert state["kv"]["podcast_queue"] == []
    assert state["ids"] == {}


# ----- all -----


@pytest.mark.parametrize("argv", [["prog", "all"], ["prog"]], ids=["explicit-all", "default-all"])
def test_all_e2e_runs_every_task_into_one_day(monkeypatch, tmp_path, argv):
    _install_newsletter_fakes(monkeypatch)
    _install_youtube_fakes(monkeypatch)
    _install_podcast_fakes(monkeypatch, tmp_path, topic="My Topic")
    _install_all_llm(
        monkeypatch, newsletter_markdown="# Daily News\n\nBody", youtube_bullet="- key point"
    )

    code = run.main(argv)

    assert code == 0
    tasks_today = _read_history(_today())["tasks"]
    assert set(tasks_today) == {"newsletter", "youtube", "podcast"}

    html = _read_index()
    newsletter_pos = html.index('<article class="task newsletter">')
    youtube_pos = html.index('<article class="task youtube">')
    podcast_pos = html.index('<article class="task podcast">')
    assert newsletter_pos < youtube_pos < podcast_pos

    state = _read_state()
    assert state["ids"].get("rss:e1") == _today()
    assert state["ids"].get("yt:vid123") == _today()
    assert state["kv"]["podcast_queue"] == []


def test_all_e2e_one_failing_task_does_not_sink_the_others(monkeypatch, tmp_path):
    _install_newsletter_fakes(monkeypatch)
    _install_youtube_fakes(monkeypatch)
    _install_podcast_fakes(monkeypatch, tmp_path, topic="My Topic")
    _install_all_llm(
        monkeypatch, newsletter_markdown="# Daily News\n\nBody", youtube_bullet="- key point"
    )
    monkeypatch.setattr(newsletter_task, "_produce", _raise(RuntimeError("boom")))

    code = run.main(["prog", "all"])

    assert code == 1  # newsletter failed -> non-zero, but the run still completes

    tasks_today = _read_history(_today())["tasks"]
    assert "newsletter" not in tasks_today
    assert "youtube" in tasks_today
    assert "podcast" in tasks_today

    state = _read_state()
    assert "rss:e1" not in state["ids"]  # failed task's items never consumed
    assert state["ids"].get("yt:vid123") == _today()
    assert state["kv"]["podcast_queue"] == []  # podcast still ran to completion
