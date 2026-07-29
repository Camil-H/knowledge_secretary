"""The two-tier cascade: the Google tier first, OpenRouter behind it. Both tiers are stubbed,
and the ledger path is confined via chdir — no real request, key or wait."""

import pytest

from src.clients import gemini, llm, openrouter
from src.core import ledger as ledger_mod
from src.core.errors import AuthError, ExternalError

# ----- test doubles -----


def _raiser(exc: Exception):
    def _raise(*_a, **_k):
        raise exc

    return _raise


def _fake_gemini(monkeypatch, text: str) -> list[dict]:
    """Replace the google tier; returns the list its calls are recorded into."""
    calls: list[dict] = []

    def _call(system, user, max_tokens, *, ledger):
        calls.append({"system": system, "user": user, "max_tokens": max_tokens, "ledger": ledger})
        return text

    monkeypatch.setattr(llm.gemini, "call", _call)
    return calls


def _fake_openrouter(monkeypatch, text: str = "from openrouter") -> list[dict]:
    calls: list[dict] = []

    def _call(system, user, max_tokens):
        calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return text

    monkeypatch.setattr(llm.openrouter, "call", _call)
    return calls


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """The ledger's write-through confined to tmp_path via cwd, with no google key or client
    left over — an unset key is how the degradation tests reach the openrouter tier."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gemini, "_CLIENT", None)
    monkeypatch.delenv(gemini.KEY_LABEL, raising=False)
    monkeypatch.setenv(openrouter.KEY_LABEL, "or-key")


# ===== Cascade =====


def test_call_prefers_the_gemini_tier(monkeypatch):
    gemini_calls = _fake_gemini(monkeypatch, "from gemini")
    openrouter_calls = _fake_openrouter(monkeypatch)

    assert llm.call("the system prompt", "the user text", max_tokens=1234) == "from gemini"
    assert openrouter_calls == []

    call = gemini_calls[0]
    assert (call["system"], call["user"], call["max_tokens"]) == (
        "the system prompt",
        "the user text",
        1234,
    )
    assert call["ledger"][ledger_mod.BUCKETS] == {}


def test_call_degrades_to_openrouter_when_the_google_tier_auth_fails(monkeypatch, caplog):
    """The live incident: an AI Studio key bound to the wrong project 403s every model, and
    that must not take the newsletter and youtube digests down with it."""
    monkeypatch.setattr(llm.gemini, "call", _raiser(AuthError(gemini.SOURCE)))
    _fake_openrouter(monkeypatch)

    with caplog.at_level("WARNING", logger=llm.logger.name):
        assert llm.call("s", "u") == "from openrouter"

    degradations = [r for r in caplog.records if "degrading to openrouter" in r.getMessage()]
    assert len(degradations) == 1


def test_call_degrades_to_openrouter_when_the_google_key_is_unset(monkeypatch):
    """No key means the real google tier raises AuthError before it dispatches anything."""
    openrouter_calls = _fake_openrouter(monkeypatch)

    assert llm.call("s", "u", max_tokens=99) == "from openrouter"
    assert openrouter_calls == [{"system": "s", "user": "u", "max_tokens": 99}]


def test_call_propagates_an_openrouter_auth_failure(monkeypatch):
    """AuthError still fails loudly at the boundary: the last independent credential is gone."""
    _fake_gemini(monkeypatch, "")
    monkeypatch.setattr(llm.openrouter, "call", _raiser(AuthError(openrouter.SOURCE)))

    with pytest.raises(AuthError):
        llm.call("s", "u")


def test_call_raises_when_every_tier_is_dry(monkeypatch):
    _fake_gemini(monkeypatch, "")
    monkeypatch.setattr(
        llm.openrouter, "call", _raiser(ExternalError("llm", detail="all models failed"))
    )

    with pytest.raises(ExternalError, match="all models failed"):
        llm.call("s", "u")
