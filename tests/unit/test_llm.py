"""OpenRouter model ranking + rate-limit backoff. Network, the model list, and
sleep are all patched, so no real API calls, keys, or waits."""

import pytest

import src.core.llm as llm

# ----- test doubles -----


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _Msg:
    def __init__(self, content):
        self.content = content


class _Completion:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _Msg(content)})()]


class _RateErr(Exception):
    status_code = 429


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


def _patch_models(monkeypatch):
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: _FakeResp(_MODELS))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)


def _cfg(primary):
    return {"models": {"summarize": {"primary": primary, "openrouter_free": False}}}


# ----- model ranking -----


def test_free_filter_excludes_paid(monkeypatch):
    _patch_models(monkeypatch)
    assert all("paid" not in m for m in llm._free_openrouter_models(llm.RANK_CONTEXT))


def test_podcast_ranks_by_output_tokens(monkeypatch):
    _patch_models(monkeypatch)
    assert llm._free_openrouter_models(llm.RANK_OUTPUT)[0] == "openrouter/big-out"


def test_summarize_ranks_by_context(monkeypatch):
    _patch_models(monkeypatch)
    assert llm._free_openrouter_models(llm.RANK_CONTEXT)[0] == "openrouter/big-ctx"


def test_resolve_prepends_primaries(monkeypatch):
    _patch_models(monkeypatch)
    cfg = {"models": {"podcast": {"primary": ["openrouter/manual:free"], "openrouter_free": True}}}
    got = llm.resolve_models("podcast", cfg)
    assert got[0] == "openrouter/manual:free"
    assert "openrouter/big-out" in got


# ----- completion: rate-limit backoff + fall-through -----


def test_call_retries_same_model_on_rate_limit(monkeypatch):
    n = {"i": 0}

    def fake_completion(model, messages, max_tokens=None):
        n["i"] += 1
        if n["i"] == 1:
            raise _RateErr("rate limit exceeded")
        return _Completion("ok")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    assert llm.call("summarize", "s", "u", _cfg(["openrouter/a:free"])) == "ok"
    assert n["i"] == 2  # 429 -> backoff -> retried the same model


def test_call_falls_through_to_next_model_on_other_error(monkeypatch):
    def fake_completion(model, messages, max_tokens=None):
        if model == "openrouter/a:free":
            raise ValueError("boom")
        return _Completion("second")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    cfg = _cfg(["openrouter/a:free", "openrouter/b:free"])
    assert llm.call("summarize", "s", "u", cfg) == "second"


def test_call_raises_when_all_candidates_fail(monkeypatch):
    def fake_completion(model, messages, max_tokens=None):
        raise ValueError("nope")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    with pytest.raises(RuntimeError, match="all models failed"):
        llm.call("summarize", "s", "u", _cfg(["openrouter/a:free"]))
