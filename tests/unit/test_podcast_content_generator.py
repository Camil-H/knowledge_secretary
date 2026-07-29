"""Grounded topic research. The shared Gemini primitive is stubbed — no request is ever made,
and the real `types` builders are used so a schema change would surface here."""

import logging

import pytest

from src.clients import gemini
from src.core import ledger as ledger_mod
from src.core.errors import AuthError, ExternalError, QuotaExhausted
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


def _stub_primitive(monkeypatch, *responses):
    """Replace gemini.generate with a scripted sequence (the last entry repeats); an Exception
    entry is raised. Returns the list its calls are recorded into."""
    calls = []
    script = list(responses)

    def _generate(model, contents, config, *, ledger):
        calls.append({"model": model, "contents": contents, "config": config, "ledger": ledger})
        item = script[min(len(calls) - 1, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(content_generator.gemini, "generate", _generate)
    return calls


def _search_models():
    return [m for m in gemini.TEXT_MODELS if m.search]


# ----- research -----


def test_research_returns_the_response_text(monkeypatch):
    _stub_primitive(monkeypatch, _Response("an overview"))
    assert research("PROTACs") == "an overview"


@pytest.mark.parametrize("text", [None, ""], ids=["none", "empty"])
def test_research_returns_empty_string_when_the_model_returns_no_text(monkeypatch, text):
    """The caller treats "" as "no material", so None must not leak out as a value."""
    _stub_primitive(monkeypatch, _Response(text))
    assert research("PROTACs") == ""


def test_research_delegates_to_the_first_search_model_with_the_topic_as_input(monkeypatch):
    calls = _stub_primitive(monkeypatch, _Response("an overview"))
    research("PROTACs")

    call = calls[0]
    assert call["model"] is _search_models()[0]
    assert call["contents"] == "PROTACs"
    assert call["config"].system_instruction == content_generator.PROMPT
    assert call["ledger"][ledger_mod.BUCKETS] == {}


def test_research_enables_the_google_search_tool(monkeypatch):
    """Without the tool the answer is ungrounded, which is the whole point of this step."""
    calls = _stub_primitive(monkeypatch, _Response("an overview"))
    research("PROTACs")

    tools = calls[0]["config"].tools
    assert [t for t in tools if t.google_search is not None]


# ----- model fallback -----


def test_research_skips_a_model_the_ledger_has_retired(monkeypatch):
    first, second = _search_models()[:2]
    ledger_mod.mark_exhausted(ledger_mod.load(), first.id)
    calls = _stub_primitive(monkeypatch, _Response("an overview"))

    assert research("PROTACs") == "an overview"
    assert [c["model"] for c in calls] == [second]


def test_research_skips_a_model_that_spent_its_daily_budget(monkeypatch):
    first, second = _search_models()[:2]
    ledger = ledger_mod.load()
    for _ in range(first.rpd):
        ledger_mod.consume(ledger, first.id)
    calls = _stub_primitive(monkeypatch, _Response("an overview"))

    research("PROTACs")
    assert [c["model"] for c in calls] == [second]


def test_research_never_tries_a_model_that_cannot_ground(monkeypatch):
    """The roster carries a high-quota model without grounding; handing it the search tool
    would fail the request, so the fallback must not reach it."""
    plain = gemini.ModelLimit("no-search", rpd=500, rpm=15, tpm=250_000, search=False)
    grounded = gemini.ModelLimit("with-search", rpd=20, rpm=5, tpm=250_000, search=True)
    monkeypatch.setattr(content_generator.gemini, "TEXT_MODELS", [plain, grounded])
    calls = _stub_primitive(monkeypatch, _Response("an overview"))

    research("PROTACs")
    assert [c["model"] for c in calls] == [grounded]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(QuotaExhausted("gemini-x"), id="quota_exhausted"),
        pytest.param(ExternalError(gemini.SOURCE), id="external_error"),
        pytest.param(_Response(""), id="empty_text"),
        pytest.param(_Response(None), id="no_text"),
        pytest.param(_Response("   "), id="blank_text"),
    ],
)
def test_research_advances_to_the_next_search_model(monkeypatch, failure):
    calls = _stub_primitive(monkeypatch, failure, _Response("a later overview"))

    assert research("PROTACs") == "a later overview"
    assert [c["model"] for c in calls] == _search_models()[:2]


def test_research_returns_empty_when_every_search_model_is_spent(monkeypatch):
    ledger = ledger_mod.load()
    for model in _search_models():
        ledger_mod.mark_exhausted(ledger, model.id)
    calls = _stub_primitive(monkeypatch, _Response("an overview"))

    assert research("PROTACs") == ""
    assert calls == []


def test_research_propagates_an_auth_error(monkeypatch):
    """Every candidate shares one key, so a credential failure is not something to fall past."""
    boom = AuthError(gemini.SOURCE)
    calls = _stub_primitive(monkeypatch, boom)

    with pytest.raises(AuthError) as ei:
        research("PROTACs")
    assert ei.value is boom
    assert len(calls) == 1


def test_research_propagates_an_untyped_primitive_failure(monkeypatch):
    """The task layer decides whether a failed episode is tolerable, not this primitive."""
    boom = RuntimeError("403 API_KEY_SERVICE_BLOCKED")
    _stub_primitive(monkeypatch, boom)

    with pytest.raises(RuntimeError) as ei:
        research("PROTACs")
    assert ei.value is boom


# ----- source logging -----


@pytest.mark.parametrize(
    "uris, expected_in_log",
    [
        (["https://a.example", "https://b.example"], "https://a.example"),
        ([], None),
        (None, None),
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
