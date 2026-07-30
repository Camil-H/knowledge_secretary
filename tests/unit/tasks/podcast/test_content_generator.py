"""Grounded topic research. The shared Gemini entry point is stubbed — no request is ever made;
which models a grounded call may use, and how it logs its sources, are tested with it."""

import pytest

from src.clients import gemini
from src.core import ledger as ledger_mod
from src.core.errors import AuthError
from src.tasks.podcast import content_generator
from src.tasks.podcast.content_generator import research


@pytest.fixture(autouse=True)
def _sandbox_ledger(monkeypatch, tmp_path):
    """research() loads the day's ledger from a cwd-relative path; keep it under tmp_path."""
    monkeypatch.chdir(tmp_path)


def _stub_call(monkeypatch, result=None, raises=None):
    """Replace gemini.call with a recorder returning `result` (or raising). Returns its calls."""
    calls = []

    def _call(system, user, max_tokens=None, *, ledger, search=False):
        calls.append(
            {
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "ledger": ledger,
                "search": search,
            }
        )
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(content_generator.gemini, "call", _call)
    return calls


# ----- research -----


@pytest.mark.parametrize("result", ["an overview", ""], ids=["text", "spent_cascade"])
def test_research_returns_what_the_grounded_call_returns(monkeypatch, result):
    _stub_call(monkeypatch, result=result)
    assert research("PROTACs") == result


def test_research_asks_for_a_grounded_completion_of_the_topic(monkeypatch):
    """`search=True` is the request: grounding-capable models and the search tool, together."""
    calls = _stub_call(monkeypatch, result="an overview")

    research("PROTACs")

    assert calls == [
        {
            "system": content_generator.PROMPT,
            "user": "PROTACs",
            "max_tokens": None,
            "ledger": ledger_mod.load(),
            "search": True,
        }
    ]


@pytest.mark.parametrize(
    "boom",
    [
        pytest.param(AuthError(gemini.SOURCE), id="auth_error"),
        pytest.param(RuntimeError("403 API_KEY_SERVICE_BLOCKED"), id="untyped"),
    ],
)
def test_research_propagates_a_failed_cascade(monkeypatch, boom):
    """The task layer decides whether a failed episode is tolerable, not this step."""
    _stub_call(monkeypatch, raises=boom)

    with pytest.raises(type(boom)) as ei:
        research("PROTACs")
    assert ei.value is boom
