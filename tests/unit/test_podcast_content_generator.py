"""Grounded topic research. The Gemini SDK is faked at the client boundary — no request is
ever made, and the real `types` builders are used so a schema change would surface here."""

import logging

import pytest

from src.tasks.podcast import content_generator
from src.tasks.podcast.content_generator import MODEL, research


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


class _RecordingModels:
    """Stands in for client.models, capturing the generate_content call."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self._response


def _fake_genai(monkeypatch, response):
    """Replace the SDK module in our namespace; returns the recording models stub."""
    models = _RecordingModels(response)

    class _Client:
        def __init__(self, *, api_key):
            self.api_key = api_key
            self.models = models

    monkeypatch.setattr(content_generator, "genai", type("genai", (), {"Client": _Client}))
    return models


# ----- research -----


def test_research_returns_the_response_text(monkeypatch):
    _fake_genai(monkeypatch, _Response("an overview"))
    assert research("PROTACs", api_key="k") == "an overview"


@pytest.mark.parametrize("text", [None, ""], ids=["none", "empty"])
def test_research_returns_empty_string_when_the_model_returns_no_text(monkeypatch, text):
    """The caller treats "" as "no material", so None must not leak out as a value."""
    _fake_genai(monkeypatch, _Response(text))
    assert research("PROTACs", api_key="k") == ""


def test_research_requests_the_search_grounded_model_with_the_topic_as_input(monkeypatch):
    models = _fake_genai(monkeypatch, _Response("an overview"))
    research("PROTACs", api_key="k")

    call = models.calls[0]
    assert call["model"] == MODEL
    assert call["contents"] == "PROTACs"  # topic is the input; the prompt is the instruction
    assert call["config"].system_instruction == content_generator.PROMPT


def test_research_enables_the_google_search_tool(monkeypatch):
    """Without the tool the answer is ungrounded, which is the whole point of this step."""
    models = _fake_genai(monkeypatch, _Response("an overview"))
    research("PROTACs", api_key="k")

    tools = models.calls[0]["config"].tools
    assert [t for t in tools if t.google_search is not None]


def test_research_passes_the_api_key_to_the_client(monkeypatch):
    """genai.Client() would otherwise read GEMINI_API_KEY, which holds the Cloud-TTS key."""
    seen = {}

    class _Client:
        def __init__(self, *, api_key):
            seen["api_key"] = api_key
            self.models = _RecordingModels(_Response("an overview"))

    monkeypatch.setattr(content_generator, "genai", type("genai", (), {"Client": _Client}))
    research("PROTACs", api_key="the-key")
    assert seen["api_key"] == "the-key"


def test_research_propagates_sdk_failures(monkeypatch):
    """The task layer decides whether a failed episode is tolerable, not this primitive."""
    boom = RuntimeError("403 API_KEY_SERVICE_BLOCKED")

    class _Models:
        def generate_content(self, **kwargs):
            raise boom

    class _Client:
        def __init__(self, *, api_key):
            self.models = _Models()

    monkeypatch.setattr(content_generator, "genai", type("genai", (), {"Client": _Client}))
    with pytest.raises(RuntimeError) as ei:
        research("PROTACs", api_key="k")
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
    _fake_genai(monkeypatch, _Response("an overview", uris=uris))
    with caplog.at_level(logging.INFO, logger=content_generator.logger.name):
        research("PROTACs", api_key="k")

    grounded = [r.getMessage() for r in caplog.records if "grounded in" in r.getMessage()]
    if expected_in_log:
        assert any(expected_in_log in message for message in grounded)
    else:
        assert grounded == []
