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


# ----- model ranking -----


def test_free_filter_excludes_paid(monkeypatch):
    _patch_models(monkeypatch)
    assert all("paid" not in m for m in llm._free_openrouter_models(llm.RANK_CONTEXT))


def test_resolve_ranks_by_tier(monkeypatch):
    _patch_models(monkeypatch)
    assert llm.resolve_models("podcast")[0] == "openrouter/big-out"  # output-tokens win
    assert llm.resolve_models("summarize")[0] == "openrouter/big-ctx"  # context wins


# ----- completion: rate-limit backoff + fall-through -----


def test_call_retries_same_model_on_rate_limit(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda task: ["openrouter/a:free"])
    n = {"i": 0}

    def fake_completion(model, messages, max_tokens=None):
        n["i"] += 1
        if n["i"] == 1:
            raise _RateErr("rate limit exceeded")
        return _Completion("ok")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    assert llm.call("summarize", "s", "u") == "ok"
    assert n["i"] == 2  # 429 -> backoff -> retried the same model


def test_call_falls_through_to_next_model_on_other_error(monkeypatch):
    monkeypatch.setattr(
        llm, "resolve_models", lambda task: ["openrouter/a:free", "openrouter/b:free"]
    )

    def fake_completion(model, messages, max_tokens=None):
        if model == "openrouter/a:free":
            raise ValueError("boom")
        return _Completion("second")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    assert llm.call("summarize", "s", "u") == "second"


def test_call_uses_fallback_when_no_models_resolve(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda task: [])
    seen = {}

    def fake_completion(model, messages, max_tokens=None):
        seen["model"] = model
        return _Completion("ok")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    assert llm.call("summarize", "s", "u") == "ok"
    assert seen["model"] == llm.FALLBACK_MODEL


def test_call_raises_when_all_candidates_fail(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda task: ["openrouter/a:free"])

    def fake_completion(model, messages, max_tokens=None):
        raise ValueError("nope")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    with pytest.raises(RuntimeError, match="all models failed"):
        llm.call("summarize", "s", "u")
