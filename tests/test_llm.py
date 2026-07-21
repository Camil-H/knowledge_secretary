"""Ranking logic for the OpenRouter-free fallback. Network is monkeypatched, so
no real API calls or keys are needed."""

import src.core.llm as llm


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


_MODELS = {
    "data": [
        {
            "id": "big-ctx",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 200000,
            "top_provider": {"max_completion_tokens": 4000},
        },
        {
            "id": "big-out",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 32000,
            "top_provider": {"max_completion_tokens": 64000},
        },
        {
            "id": "paid",
            "pricing": {"prompt": "0.001", "completion": "0.002"},
            "context_length": 1000000,
            "top_provider": {"max_completion_tokens": 99999},
        },
    ]
}


def _patch(monkeypatch):
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: _FakeResp(_MODELS))


def test_free_filter_excludes_paid(monkeypatch):
    _patch(monkeypatch)
    got = llm._free_openrouter_models(llm.RANK_CONTEXT)
    assert all("paid" not in m for m in got)


def test_podcast_ranks_by_output_tokens(monkeypatch):
    _patch(monkeypatch)
    got = llm._free_openrouter_models(llm.RANK_OUTPUT)
    assert got[0] == "openrouter/big-out"  # 64k output wins for podcast


def test_summarize_ranks_by_context(monkeypatch):
    _patch(monkeypatch)
    got = llm._free_openrouter_models(llm.RANK_CONTEXT)
    assert got[0] == "openrouter/big-ctx"  # 200k context wins otherwise


def test_resolve_prepends_primaries(monkeypatch):
    _patch(monkeypatch)
    cfg = {"models": {"podcast": {"primary": ["gemini/gemini-2.5-flash"], "openrouter_free": True}}}
    got = llm.resolve_models("podcast", cfg)
    assert got[0] == "gemini/gemini-2.5-flash"
    assert "openrouter/big-out" in got
