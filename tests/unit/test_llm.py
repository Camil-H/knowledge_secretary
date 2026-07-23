"""OpenRouter model ranking + rate-limit backoff. Network, the model list, and
sleep are all patched, so no real API calls, keys, or waits."""

import httpx
import pytest

import src.core.llm as llm
from src.core.errors import AuthError, ExternalError

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


class _StatusErr(Exception):
    """Generic error carrying an optional status_code, for _is_rate_limit/_is_auth matrices."""

    def __init__(self, msg: str, *, status_code: int | None = None) -> None:
        super().__init__(msg)
        self.status_code = status_code


def _raiser(exc: Exception):
    def _raise(*_a, **_k):
        raise exc

    return _raise


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


@pytest.fixture(autouse=True)
def _reset_model_cache():
    llm._reset_model_cache()
    yield
    llm._reset_model_cache()


# ----- model ranking -----


def test_free_filter_excludes_paid(monkeypatch):
    _patch_models(monkeypatch)
    assert all("paid" not in m for m in llm._free_openrouter_models(llm.RANK_CONTEXT))


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "google/lyria-3-pro-preview", "context_length": 1000000},
        {"id": "nvidia/x-content-safety:free", "context_length": 1000000},
        {"id": "openrouter/free", "context_length": 1000000},
        {
            "id": "x/music:free",
            "context_length": 1000000,
            "architecture": {"output_modalities": ["audio"]},
        },
    ],
    ids=["music", "guardrail", "router", "audio-output"],
)
def test_free_filter_excludes_non_text_writers(monkeypatch, bad):
    bad = {**bad, "pricing": {"prompt": "0", "completion": "0"}}
    good = {
        "id": "good/writer",
        "pricing": {"prompt": "0", "completion": "0"},
        "context_length": 1000,
    }
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: _FakeResp({"data": [bad, good]}))
    # the non-writer ranks first by its 1M context, but must be filtered out entirely
    assert llm._free_openrouter_models(llm.RANK_CONTEXT) == ["openrouter/good/writer"]


def test_resolve_ranks_by_tier(monkeypatch):
    _patch_models(monkeypatch)
    # none of _MODELS' ids are in the PREFERRED lists, so live ranking order stands
    assert llm.resolve_models(podcast=True)[0] == "openrouter/big-out"  # output-tokens win
    assert llm.resolve_models()[0] == "openrouter/big-ctx"  # context by default


def test_resolve_models_prefers_curated_ids_present_in_live_list(monkeypatch):
    preferred_first, preferred_second = llm.PREFERRED_CONTEXT[0], llm.PREFERRED_CONTEXT[1]
    models = {
        "data": [
            {
                "id": "openrouter/big-ctx".removeprefix("openrouter/"),
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 200000,
                "top_provider": {"max_completion_tokens": 4000},
            },
            {
                "id": preferred_second.removeprefix("openrouter/"),
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 500,  # would rank last on live context ranking
                "top_provider": {"max_completion_tokens": 10},
            },
            {
                "id": preferred_first.removeprefix("openrouter/"),
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 100,  # would rank last on live context ranking
                "top_provider": {"max_completion_tokens": 10},
            },
        ]
    }
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: _FakeResp(models))

    result = llm.resolve_models()

    # preferred ids present in the live list lead, in PREFERRED_CONTEXT order,
    # despite ranking last by raw context length
    assert result[:2] == [preferred_first, preferred_second]
    assert result[2] == "openrouter/big-ctx"


def test_resolve_models_skips_absent_preferred_ids_without_crashing(monkeypatch):
    _patch_models(monkeypatch)  # none of PREFERRED_CONTEXT's ids are present
    assert not any(m in llm.PREFERRED_CONTEXT for m in llm.resolve_models())
    assert llm.resolve_models() == ["openrouter/big-ctx", "openrouter/big-out"]


# ----- model ranking cache -----


def test_free_openrouter_models_memoizes_per_rank_mode(monkeypatch):
    calls = {"n": 0}

    def fake_get(*_a, **_k):
        calls["n"] += 1
        return _FakeResp(_MODELS)

    monkeypatch.setattr(llm.httpx, "get", fake_get)

    first = llm._free_openrouter_models(llm.RANK_CONTEXT)
    assert llm._free_openrouter_models(llm.RANK_CONTEXT) == first
    assert calls["n"] == 1  # second call for the same rank mode hits the cache

    llm._free_openrouter_models(llm.RANK_OUTPUT)
    assert calls["n"] == 2  # a different rank mode still fetches its own catalog


def test_free_openrouter_models_does_not_cache_a_failed_fetch(monkeypatch):
    calls = {"n": 0}

    def fake_get(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.HTTPError("boom")
        return _FakeResp(_MODELS)

    monkeypatch.setattr(llm.httpx, "get", fake_get)

    assert llm._free_openrouter_models(llm.RANK_CONTEXT) == []
    assert llm._free_openrouter_models(llm.RANK_CONTEXT) != []
    assert calls["n"] == 2  # the failed first fetch wasn't cached, so it retried


# ----- model list degradation -----


@pytest.mark.parametrize(
    "get_stub",
    [
        pytest.param(_raiser(httpx.HTTPError("boom")), id="http_error"),
        pytest.param(lambda *a, **k: _FakeResp({"unexpected": []}), id="missing_data_key"),
    ],
)
def test_free_openrouter_models_degrades_to_empty(monkeypatch, get_stub):
    monkeypatch.setattr(llm.httpx, "get", get_stub)
    assert llm._free_openrouter_models(llm.RANK_CONTEXT) == []


def test_call_reaches_fallback_when_model_list_degrades(monkeypatch):
    monkeypatch.setattr(llm.httpx, "get", _raiser(httpx.HTTPError("boom")))
    seen = {}

    def fake_completion(model, messages, max_tokens=None):
        seen["model"] = model
        return _Completion("ok")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    assert llm.call("s", "u") == "ok"
    assert seen["model"] == llm.FALLBACK_MODEL


# ----- rank key edge cases -----

_EDGE_MODELS = {
    "data": [
        {
            "id": "no-provider",
            "pricing": {"prompt": "0", "completion": "0"},
            "top_provider": None,
            "context_length": 5000,
        },
        {
            "id": "no-context",
            "pricing": {"prompt": "0", "completion": "0"},
            "top_provider": {"max_completion_tokens": 10},
            # context_length intentionally absent
        },
    ]
}


@pytest.mark.parametrize(
    "rank, expected_first",
    [
        # top_provider=None -> ranks 0, loses to the 10-token model
        (llm.RANK_OUTPUT, "openrouter/no-context"),
        # missing context_length -> ranks 0, loses to the 5000-context model
        (llm.RANK_CONTEXT, "openrouter/no-provider"),
    ],
)
def test_free_openrouter_models_rank_key_handles_missing_fields(monkeypatch, rank, expected_first):
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: _FakeResp(_EDGE_MODELS))
    result = llm._free_openrouter_models(rank)
    assert result[0] == expected_first


# ----- completion: rate-limit backoff + fall-through -----


def test_call_retries_same_model_on_rate_limit(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda: ["openrouter/a:free"])
    n = {"i": 0}

    def fake_completion(model, messages, max_tokens=None):
        n["i"] += 1
        if n["i"] == 1:
            raise _RateErr("rate limit exceeded")
        return _Completion("ok")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    assert llm.call("s", "u") == "ok"
    assert n["i"] == 2  # 429 -> backoff -> retried the same model


def test_call_falls_through_to_next_model_on_other_error(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda: ["openrouter/a:free", "openrouter/b:free"])

    def fake_completion(model, messages, max_tokens=None):
        if model == "openrouter/a:free":
            raise ValueError("boom")
        return _Completion("second")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    assert llm.call("s", "u") == "second"


def test_call_uses_fallback_when_no_models_resolve(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda: [])
    seen = {}

    def fake_completion(model, messages, max_tokens=None):
        seen["model"] = model
        return _Completion("ok")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    assert llm.call("s", "u") == "ok"
    assert seen["model"] == llm.FALLBACK_MODEL


def test_call_raises_when_all_candidates_fail(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda: ["openrouter/a:free"])

    def fake_completion(model, messages, max_tokens=None):
        raise ValueError("nope")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    with pytest.raises(ExternalError, match="all models failed"):
        llm.call("s", "u")


def test_call_raises_auth_error_immediately(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda: ["openrouter/a:free", "openrouter/b:free"])
    tried = []

    def fake_completion(model, messages, max_tokens=None):
        tried.append(model)
        raise ValueError("No user or org id found in auth cookie")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    with pytest.raises(AuthError):
        llm.call("s", "u")
    assert tried == ["openrouter/a:free"]  # auth fails loudly on the first model, no fallback


def test_call_external_error_carries_last_exception_as_cause(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda: ["openrouter/a:free"])
    boom = ValueError("nope")

    monkeypatch.setattr(llm.litellm, "completion", _raiser(boom))
    with pytest.raises(ExternalError) as exc_info:
        llm.call("s", "u")
    assert exc_info.value.cause is boom


# ----- completion: persistent rate limit exhausts retries -----


@pytest.mark.parametrize(
    "models, expect_next_model",
    [
        pytest.param(["openrouter/a:free"], False, id="single_model_raises"),
        pytest.param(["openrouter/a:free", "openrouter/b:free"], True, id="advances_to_next_model"),
    ],
)
def test_call_persistent_rate_limit_exhausts_retries(monkeypatch, models, expect_next_model):
    monkeypatch.setattr(llm, "resolve_models", lambda: models)
    calls = {"a": 0}

    def fake_completion(model, messages, max_tokens=None):
        if model == "openrouter/a:free":
            calls["a"] += 1
            raise _RateErr("rate limit exceeded")
        return _Completion("ok")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)

    if expect_next_model:
        assert llm.call("s", "u") == "ok"
    else:
        with pytest.raises(ExternalError):
            llm.call("s", "u")

    # count derived from the retry constant, not a hardcoded literal
    assert calls["a"] == llm._RATE_LIMIT_RETRIES


def test_call_backoff_doubles_and_caps(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda: ["openrouter/a:free"])
    retries = 7  # enough attempts for the doubling sequence to actually hit the cap
    monkeypatch.setattr(llm, "_RATE_LIMIT_RETRIES", retries)
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(llm.litellm, "completion", _raiser(_RateErr("rate limit exceeded")))

    with pytest.raises(ExternalError):
        llm.call("s", "u")

    expected = []
    backoff = llm._BACKOFF_START_S
    for _ in range(retries - 1):  # one sleep per retried attempt, none after the last
        expected.append(backoff)
        backoff = min(backoff * 2, llm._BACKOFF_CAP_S)
    assert sleeps == expected


# ----- completion: empty / whitespace content -----


@pytest.mark.parametrize("empty_content", ["", "   ", "\n\t "])
def test_call_falls_through_on_empty_or_whitespace_content(monkeypatch, empty_content):
    monkeypatch.setattr(llm, "resolve_models", lambda: ["openrouter/a:free", "openrouter/b:free"])

    def fake_completion(model, messages, max_tokens=None):
        if model == "openrouter/a:free":
            return _Completion(empty_content)
        return _Completion("second")

    monkeypatch.setattr(llm.litellm, "completion", fake_completion)
    assert llm.call("s", "u") == "second"


def test_call_raises_when_all_models_return_empty(monkeypatch):
    monkeypatch.setattr(llm, "resolve_models", lambda: ["openrouter/a:free", "openrouter/b:free"])
    monkeypatch.setattr(
        llm.litellm, "completion", lambda model, messages, max_tokens=None: _Completion("   ")
    )
    with pytest.raises(ExternalError, match="all models failed"):
        llm.call("s", "u")


# ----- _is_rate_limit / _is_auth -----


@pytest.mark.parametrize(
    "exc, expected",
    [
        pytest.param(_StatusErr("boom", status_code=429), True, id="status_code_429"),
        pytest.param(ValueError("Rate limit exceeded, try later"), True, id="message_substring"),
        pytest.param(ValueError("totally unrelated"), False, id="negative"),
    ],
)
def test_is_rate_limit(exc, expected):
    assert llm._is_rate_limit(exc) is expected


@pytest.mark.parametrize(
    "exc, expected",
    [
        pytest.param(_StatusErr("boom", status_code=401), True, id="status_code_401"),
        pytest.param(
            ValueError("No user or org id found in auth cookie"),
            True,
            id="no_user_or_org_substring",
        ),
        pytest.param(ValueError("invalid API key provided"), True, id="invalid_api_key"),
        pytest.param(ValueError("totally unrelated"), False, id="negative"),
        pytest.param(
            ValueError("Unknown author, please retry"), False, id="author_substring_not_auth"
        ),
    ],
)
def test_is_auth(exc, expected):
    assert llm._is_auth(exc) is expected
