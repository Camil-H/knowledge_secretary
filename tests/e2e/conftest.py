"""Shared e2e sandbox: fakes every external boundary the pipeline touches (RSS,
YouTube, LLM, Cloud TTS, gh) and confines state/history/output under tmp_path.

`run.main([...])` is driven for real — only the collaborators at the network/
subprocess/LLM edge are stubbed, so the registry/gather/produce/deliver wiring
underneath is genuinely exercised.
"""

import json
import os
from datetime import UTC, datetime

import pytest

from src.clients import llm
from src.delivery import site
from src.fetchers import rss
from src.fetchers import youtube as yt
from src.tasks.newsletter import task as newsletter_task
from src.tasks.podcast import task as podcast_task
from src.tasks.youtube import task as youtube_task

# == Fixtures ==================================================================


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Confine state/seen.json (via cwd — its path default is bound at def time,
    see plan note) and history/public (via site's module globals) under tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(site, "HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setattr(site, "OUT_DIR", str(tmp_path / "public"))
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)


# == Read helpers ==============================================================


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _read_state() -> dict:
    with open("state/seen.json") as f:
        return json.load(f)


def _read_history(date: str) -> dict:
    with open(os.path.join(site.HISTORY_DIR, f"{date}.json")) as f:
        return json.load(f)


def _render_index() -> str:
    """Render the page as the publish action does, once the run's history is committed."""
    site.render()
    with open(os.path.join(site.OUT_DIR, "index.html")) as f:
        return f.read()


# == Fake doubles ==============================================================


class _CompletedOK:
    """Stand-in for subprocess.CompletedProcess on the happy path."""

    returncode = 0
    stdout = "[]"  # empty release list for the prune step; ignored by create/upload
    stderr = ""


def _raise(exc: Exception):
    """A collaborator stub that raises `exc` unconditionally."""

    def _fn(*args, **kwargs):
        raise exc

    return _fn


# == Fake factories / installers ==============================================


def _install_newsletter_fakes(monkeypatch, *, markdown: str = "# Daily News\n\nBody"):
    monkeypatch.setattr(
        newsletter_task,
        "SOURCES",
        [{"key": "blog", "kind": "feed", "section": "Blogs", "url": "http://feed"}],
    )
    monkeypatch.setattr(
        rss,
        "fetch",
        lambda url: {
            "title": "Blog",
            "entries": [
                {
                    "id": "e1",
                    "title": "Post A",
                    "link": "http://feed/a",
                    "published": datetime.now(UTC),
                    "summary": "Body A",
                    "raw": {},
                }
            ],
        },
    )
    monkeypatch.setattr(llm, "call", lambda system, user, max_tokens=None: markdown)


def _install_youtube_fakes(monkeypatch, *, bullet: str = "- key point"):
    monkeypatch.setattr(
        youtube_task,
        "SOURCES",
        [
            {
                "key": "yt_a",
                "kind": "yt_channel",
                "section": "Science",
                "channel_id": "C1",
                "enrich": ["transcript"],
            }
        ],
    )
    monkeypatch.setattr(
        yt,
        "channel_videos",
        lambda cid: {
            "channel": "ChanName",
            "videos": [
                {
                    "video_id": "vid123",
                    "title": "Vid A",
                    "url": "http://yt/vid123",
                    "published": datetime.now(UTC),
                    "summary": "",
                }
            ],
        },
    )
    monkeypatch.setattr(yt, "transcript", lambda vid: "transcript text")
    monkeypatch.setattr(llm, "call", lambda system, user, max_tokens=None: bullet)


def _install_podcast_fakes(monkeypatch, tmp_path, *, topic: str = "My Topic"):
    """Fakes the podcast boundary; returns the list `gh` invocations are recorded into."""
    monkeypatch.setattr(podcast_task, "TOPICS", [topic])
    monkeypatch.setattr(
        podcast_task.content_generator, "research", lambda topic: _PODCAST_RESEARCH_TEXT
    )
    monkeypatch.setattr(
        podcast_task.transcript,
        "generate",
        lambda topic, research, *, call: _PODCAST_TRANSCRIPT,
    )

    ep = tmp_path / "ep.mp3"
    ep.write_bytes(b"\x00")

    def _fake_synthesize(transcript, out_path, *, ledger):
        assert transcript  # the audio layer must receive our transcript, not regenerate one
        return str(ep)

    monkeypatch.setattr(podcast_task.audio, "synthesize", _fake_synthesize)
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")

    calls: list[list[str]] = []

    def _fake_run(argv, **kw):
        calls.append(argv)
        return _CompletedOK()

    monkeypatch.setattr(site.subprocess, "run", _fake_run)
    return calls


def _install_all_llm(monkeypatch, *, newsletter_markdown: str, youtube_bullet: str):
    """One llm.call fake keyed on `system` so newsletter + youtube can be exercised in the
    same run without clobbering each other's stub. The podcast no longer routes through
    llm.call — its research goes straight to content_generator."""

    def _call(system, user, max_tokens=None):
        if system == newsletter_task.EDITOR_PROMPT:
            return newsletter_markdown
        return youtube_bullet

    monkeypatch.setattr(llm, "call", _call)


_PODCAST_RESEARCH_TEXT = "search-grounded overview of the topic"
_PODCAST_TRANSCRIPT = "<Person1>Hello there.</Person1>\n<Person2>Glad to be here.</Person2>"
