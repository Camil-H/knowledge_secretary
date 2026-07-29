"""Grounded topic research. The shared Gemini primitive is stubbed — no request is ever made,
and the real `types` builders are used so a schema change would surface here."""

import logging

import pytest

from src.core import gemini
from src.tasks.podcast import content_generator
from src.tasks.podcast.content_generator import research


class _Web:
    def __init__(self, uri):
        self.uri = uri


class _Chunk:
    def __init__(self, uri):
        self.web = _Web(uri) if uri else None


class _Metadata:
    def __init__(self, uris):
        self.grounding_chunks = [_Chunk(u) for u in uris]


class _Candidate:
    def __init__(self, uris):
        self.grounding_metadata = _Metadata(uris) if uris is not None else None


class _Response:
    def __init__(self, text, uris=None):
        self.text = text
        self.candidates = [_Candidate(uris)]


@pytest.fixture(autouse=True)
def _sandbox_ledger(monkeypatch, tmp_path):
    """research() loads the day's ledger from a cwd-relative path; keep it under tmp_path."""
    monkeypatch.chdir(tmp_path)


def _stub_primitive(monkeypatch, response):
    """Replace gemini.generate; returns the list its calls are recorded into."""
    calls = []

    def _generate(model, contents, config, *, ledger):
        calls.append({"model": model, "contents": contents, "config": config, "ledger": ledger})
        return response

    monkeypatch.setattr(content_generator.gemini, "generate", _generate)
    return calls


# ----- research -----


def test_research_returns_the_response_text(monkeypatch):
    _stub_primitive(monkeypatch, _Response("an overview"))
    assert research("PROTACs") == "an overview"


@pytest.mark.parametrize("text", [None, ""], ids=["none", "empty"])
def test_research_returns_empty_string_when_the_model_returns_no_text(monkeypatch, text):
    """The caller treats "" as "no material", so None must not leak out as a value."""
    _stub_primitive(monkeypatch, _Response(text))
    assert research("PROTACs") == ""


def test_research_delegates_to_the_first_gemini_model_with_the_topic_as_input(monkeypatch):
    calls = _stub_primitive(monkeypatch, _Response("an overview"))
    research("PROTACs")

    call = calls[0]
    assert call["model"] is gemini.TEXT_MODELS[0]
    assert call["contents"] == "PROTACs"  # topic is the input; the prompt is the instruction
    assert call["config"].system_instruction == content_generator.PROMPT
    assert call["ledger"]["models"] == {}  # today's shared ledger, so research shares the budget


def test_research_enables_the_google_search_tool(monkeypatch):
    """Without the tool the answer is ungrounded, which is the whole point of this step."""
    calls = _stub_primitive(monkeypatch, _Response("an overview"))
    research("PROTACs")

    tools = calls[0]["config"].tools
    assert [t for t in tools if t.google_search is not None]


def test_research_propagates_primitive_failures(monkeypatch):
    """The task layer decides whether a failed episode is tolerable, not this primitive."""
    boom = RuntimeError("403 API_KEY_SERVICE_BLOCKED")

    def _generate(*_a, **_k):
        raise boom

    monkeypatch.setattr(content_generator.gemini, "generate", _generate)
    with pytest.raises(RuntimeError) as ei:
        research("PROTACs")
    assert ei.value is boom


# ----- source logging -----


@pytest.mark.parametrize(
    "uris, expected_in_log",
    [
        (["https://a.example", "https://b.example"], "https://a.example"),
        ([], None),
        (None, None),  # no grounding metadata at all
    ],
    ids=["logs_uris", "no_chunks", "no_metadata"],
)
def test_research_logs_the_grounded_sources(monkeypatch, caplog, uris, expected_in_log):
    """The grounded URIs are the only record of what an episode was built from."""
    _stub_primitive(monkeypatch, _Response("an overview", uris=uris))
    with caplog.at_level(logging.INFO, logger=content_generator.logger.name):
        research("PROTACs")

    grounded = [r.getMessage() for r in caplog.records if "grounded in" in r.getMessage()]
    if expected_in_log:
        assert any(expected_in_log in message for message in grounded)
    else:
        assert grounded == []
