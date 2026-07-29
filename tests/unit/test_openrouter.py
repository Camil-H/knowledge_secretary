"""The OpenRouter tier: its model catalog, one chat completion, and the deadline-bounded loop
over the ranked candidates. httpx and sleep are faked — no real request, key or wait."""

import httpx
import pytest

import src.core.openrouter as openrouter
from src.core.errors import AuthError, ExternalError

# ----- test doubles -----


class _FakeResp:
    """An httpx-shaped response for both the catalog fetch and a chat completion."""

    def __init__(self, payload, *, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class _StatusErr(Exception):
    """Generic error carrying an optional status_code, for _is_rate_limit/_is_auth matrices."""

    def __init__(self, msg: str, *, status_code: int | None = None) -> None:
        super().__init__(msg)
        self.status_code = status_code


def _raiser(exc: Exception):
    def _raise(*_a, **_k):
        raise exc

    return _raise


def _fake_clock(monkeypatch, start: float = 0.0):
    """Deterministic monotonic clock; sleep advances it so backoff spends the budget."""
    clock = {"t": start}
    monkeypatch.setattr(openrouter.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    return clock


def _completing(text: str):
    def _complete(model, messages, max_tokens):
        return text

    return _complete


_MODELS = {
    "data": [
        {
            "id": "big-ctx",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 200000,
        },
        {
            "id": "small-ctx",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 32000,
        },
        {
            "id": "paid",
            "pricing": {"prompt": "0.001", "completion": "0.002"},
            "context_length": 1000000,
        },
    ]
}


def _patch_models(monkeypatch):
    monkeypatch.setattr(openrouter.httpx, "get", lambda *a, **k: _FakeResp(_MODELS))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No sleeps and no catalog memoized between tests."""
    monkeypatch.setattr(openrouter.time, "sleep", lambda _s: None)
    monkeypatch.setenv(openrouter.KEY_LABEL, "or-key")
    openrouter._reset_model_cache()
    yield
    openrouter._reset_model_cache()


# ===== Tier =====

# ----- rate-limit backoff + fall-through -----


def test_call_retries_the_same_model_on_rate_limit(monkeypatch):
    monkeypatch.setattr(openrouter, "models", lambda: ["openrouter/a:free"])
    n = {"i": 0}

    def _complete(model, messages, max_tokens):
        n["i"] += 1
        if n["i"] == 1:
            raise _StatusErr("rate limit exceeded", status_code=429)
        return "ok"

    monkeypatch.setattr(openrouter, "complete", _complete)
    assert openrouter.call("s", "u", None) == "ok"
    assert n["i"] == 2


def test_call_falls_through_to_the_next_model_on_other_error(monkeypatch):
    monkeypatch.setattr(openrouter, "models", lambda: ["openrouter/a:free", "openrouter/b:free"])

    def _complete(model, messages, max_tokens):
        if model == "openrouter/a:free":
            raise ValueError("boom")
        return "second"

    monkeypatch.setattr(openrouter, "complete", _complete)
    assert openrouter.call("s", "u", None) == "second"


@pytest.mark.parametrize("candidates", [[], None], ids=["empty_list", "degraded_fetch"])
def test_call_uses_the_fallback_model_when_none_resolve(monkeypatch, candidates):
    if candidates is None:
        monkeypatch.setattr(openrouter.httpx, "get", _raiser(httpx.HTTPError("boom")))
    else:
        monkeypatch.setattr(openrouter, "models", lambda: candidates)
    seen = {}

    def _complete(model, messages, max_tokens):
        seen["model"] = model
        return "ok"

    monkeypatch.setattr(openrouter, "complete", _complete)
    assert openrouter.call("s", "u", None) == "ok"
    assert seen["model"] == openrouter.FALLBACK_MODEL


def test_call_raises_auth_error_immediately(monkeypatch):
    monkeypatch.setattr(openrouter, "models", lambda: ["openrouter/a:free", "openrouter/b:free"])
    tried = []

    def _complete(model, messages, max_tokens):
        tried.append(model)
        raise ValueError("No user or org id found in auth cookie")

    monkeypatch.setattr(openrouter, "complete", _complete)
    with pytest.raises(AuthError):
        openrouter.call("s", "u", None)
    assert tried == ["openrouter/a:free"]


def test_call_external_error_carries_the_last_exception_as_cause(monkeypatch):
    monkeypatch.setattr(openrouter, "models", lambda: ["openrouter/a:free"])
    boom = ValueError("nope")
    monkeypatch.setattr(openrouter, "complete", _raiser(boom))

    with pytest.raises(ExternalError) as ei:
        openrouter.call("s", "u", None)
    assert ei.value.cause is boom


@pytest.mark.parametrize(
    "candidates, expect_next_model",
    [
        pytest.param(["openrouter/a:free"], False, id="single_model_raises"),
        pytest.param(["openrouter/a:free", "openrouter/b:free"], True, id="advances_to_next_model"),
    ],
)
def test_call_persistent_rate_limit_exhausts_retries(monkeypatch, candidates, expect_next_model):
    monkeypatch.setattr(openrouter, "models", lambda: candidates)
    calls = {"a": 0}

    def _complete(model, messages, max_tokens):
        if model == "openrouter/a:free":
            calls["a"] += 1
            raise _StatusErr("rate limit exceeded", status_code=429)
        return "ok"

    monkeypatch.setattr(openrouter, "complete", _complete)

    if expect_next_model:
        assert openrouter.call("s", "u", None) == "ok"
    else:
        with pytest.raises(ExternalError):
            openrouter.call("s", "u", None)

    assert calls["a"] == openrouter._RATE_LIMIT_RETRIES


def test_call_backoff_doubles_and_caps(monkeypatch):
    monkeypatch.setattr(openrouter, "models", lambda: ["openrouter/a:free"])
    retries = 7
    monkeypatch.setattr(openrouter, "_RATE_LIMIT_RETRIES", retries)
    sleeps: list[float] = []
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(openrouter, "complete", _raiser(_StatusErr("rate limit", status_code=429)))

    with pytest.raises(ExternalError):
        openrouter.call("s", "u", None)

    expected = []
    backoff = openrouter._BACKOFF_START_S
    for _ in range(retries - 1):
        expected.append(backoff)
        backoff = min(backoff * 2, openrouter._BACKOFF_CAP_S)
    assert sleeps == expected


# ----- deadline / budget -----


def test_call_abandons_the_cascade_once_the_deadline_is_exceeded(monkeypatch):
    _fake_clock(monkeypatch)
    monkeypatch.setattr(openrouter, "_DEADLINE_S", 5.0)
    candidates = [f"openrouter/m{i}:free" for i in range(openrouter._FREE_LIMIT)]
    monkeypatch.setattr(openrouter, "models", lambda: candidates)
    tried = []

    def _complete(model, messages, max_tokens):
        tried.append(model)
        raise _StatusErr("rate limit exceeded", status_code=429)

    monkeypatch.setattr(openrouter, "complete", _complete)
    with pytest.raises(ExternalError, match="all models failed"):
        openrouter.call("s", "u", None)

    assert len(set(tried)) < len(candidates)
    assert len(tried) < len(candidates) * openrouter._RATE_LIMIT_RETRIES


# ----- empty / whitespace content -----


@pytest.mark.parametrize("empty_content", ["", "   ", "\n\t "])
def test_call_falls_through_on_empty_or_whitespace_content(monkeypatch, empty_content):
    monkeypatch.setattr(openrouter, "models", lambda: ["openrouter/a:free", "openrouter/b:free"])

    def _complete(model, messages, max_tokens):
        return empty_content if model == "openrouter/a:free" else "second"

    monkeypatch.setattr(openrouter, "complete", _complete)
    assert openrouter.call("s", "u", None) == "second"


def test_call_raises_when_all_models_return_empty(monkeypatch):
    monkeypatch.setattr(openrouter, "models", lambda: ["openrouter/a:free", "openrouter/b:free"])
    monkeypatch.setattr(openrouter, "complete", _completing("   "))
    with pytest.raises(ExternalError, match="all models failed"):
        openrouter.call("s", "u", None)


# ----- complete -----


def _capture_post(monkeypatch, response: _FakeResp) -> dict:
    seen: dict = {}

    def _post(url, *, headers, json, timeout):
        seen.update(url=url, headers=headers, json=json, timeout=timeout)
        return response

    monkeypatch.setattr(openrouter.httpx, "post", _post)
    return seen


def test_complete_posts_the_openai_shaped_body(monkeypatch):
    seen = _capture_post(monkeypatch, _FakeResp({"choices": [{"message": {"content": "hello"}}]}))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    assert openrouter.complete("openrouter/vendor/model:free", messages, 99) == "hello"
    assert seen["url"] == openrouter.COMPLETIONS_URL
    assert seen["json"] == {
        "model": "vendor/model:free",
        "messages": messages,
        "max_tokens": 99,
    }
    assert seen["headers"]["Authorization"].endswith("or-key")
    assert seen["timeout"] == openrouter._HTTP_TIMEOUT_S


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
    ids=["no_choices_key", "no_choices", "no_content_key", "null_content"],
)
def test_complete_returns_empty_on_a_textless_response(monkeypatch, payload):
    _capture_post(monkeypatch, _FakeResp(payload))
    assert openrouter.complete("openrouter/a:free", [], None) == ""


@pytest.mark.parametrize(
    "status, body, expected",
    [
        pytest.param(429, '{"error": "rate limit"}', openrouter._is_rate_limit, id="rate_limited"),
        pytest.param(401, '{"error": "unauthorized"}', openrouter._is_auth, id="unauthorized"),
        pytest.param(
            402, '{"error": "rate limit reached"}', openrouter._is_rate_limit, id="body_only"
        ),
    ],
)
def test_complete_raises_a_classifiable_error(monkeypatch, status, body, expected):
    _capture_post(monkeypatch, _FakeResp(None, status_code=status, text=body))

    with pytest.raises(openrouter.OpenRouterError) as ei:
        openrouter.complete("openrouter/a:free", [], None)
    assert expected(ei.value)


def test_complete_truncates_the_error_body(monkeypatch):
    _capture_post(monkeypatch, _FakeResp(None, status_code=500, text="x" * 5000))

    with pytest.raises(openrouter.OpenRouterError) as ei:
        openrouter.complete("openrouter/a:free", [], None)
    assert len(str(ei.value)) < openrouter._MAX_ERROR_BODY_CHARS * 2


def test_complete_raises_auth_error_without_a_key(monkeypatch):
    monkeypatch.delenv(openrouter.KEY_LABEL, raising=False)
    monkeypatch.setattr(openrouter.httpx, "post", _raiser(AssertionError("must not be called")))

    with pytest.raises(AuthError, match=openrouter.KEY_LABEL):
        openrouter.complete("openrouter/a:free", [], None)


# ===== Model resolution =====


def test_free_filter_excludes_paid(monkeypatch):
    _patch_models(monkeypatch)
    assert all("paid" not in m for m in openrouter._free_models())


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
    monkeypatch.setattr(openrouter.httpx, "get", lambda *a, **k: _FakeResp({"data": [bad, good]}))
    # the non-writer ranks first by its 1M context, but must be filtered out entirely
    assert openrouter._free_models() == ["openrouter/good/writer"]


def test_free_models_ranks_by_context(monkeypatch):
    _patch_models(monkeypatch)
    assert openrouter._free_models() == ["openrouter/big-ctx", "openrouter/small-ctx"]


def test_free_models_rank_key_handles_a_missing_context_length(monkeypatch):
    edge = {
        "data": [
            {"id": "no-context", "pricing": {"prompt": "0", "completion": "0"}},
            {
                "id": "has-context",
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 5000,
            },
        ]
    }
    monkeypatch.setattr(openrouter.httpx, "get", lambda *a, **k: _FakeResp(edge))
    assert openrouter._free_models()[0] == "openrouter/has-context"


def test_models_prefers_curated_ids_present_in_live_list(monkeypatch):
    preferred_first, preferred_second = (
        openrouter.PREFERRED_CONTEXT[0],
        (openrouter.PREFERRED_CONTEXT[1]),
    )
    catalog = {
        "data": [
            {
                "id": "big-ctx",
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 200000,
            },
            {
                "id": preferred_second.removeprefix("openrouter/"),
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 500,  # would rank last on live context ranking
            },
            {
                "id": preferred_first.removeprefix("openrouter/"),
                "pricing": {"prompt": "0", "completion": "0"},
                "context_length": 100,  # would rank last on live context ranking
            },
        ]
    }
    monkeypatch.setattr(openrouter.httpx, "get", lambda *a, **k: _FakeResp(catalog))

    result = openrouter.models()

    assert result[:2] == [preferred_first, preferred_second]
    assert result[2] == "openrouter/big-ctx"


def test_models_skips_absent_preferred_ids_without_crashing(monkeypatch):
    _patch_models(monkeypatch)  # none of PREFERRED_CONTEXT's ids are present
    assert openrouter.models() == ["openrouter/big-ctx", "openrouter/small-ctx"]


# ----- catalog cache + degradation -----


def test_free_models_memoizes_the_catalog(monkeypatch):
    calls = {"n": 0}

    def _get(*_a, **_k):
        calls["n"] += 1
        return _FakeResp(_MODELS)

    monkeypatch.setattr(openrouter.httpx, "get", _get)

    first = openrouter._free_models()
    assert openrouter._free_models() == first
    assert calls["n"] == 1


def test_free_models_does_not_cache_a_failed_fetch(monkeypatch):
    calls = {"n": 0}

    def _get(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.HTTPError("boom")
        return _FakeResp(_MODELS)

    monkeypatch.setattr(openrouter.httpx, "get", _get)

    assert openrouter._free_models() == []
    assert openrouter._free_models() != []
    assert calls["n"] == 2


@pytest.mark.parametrize(
    "get_stub",
    [
        pytest.param(_raiser(httpx.HTTPError("boom")), id="http_error"),
        pytest.param(lambda *a, **k: _FakeResp({"unexpected": []}), id="missing_data_key"),
    ],
)
def test_free_models_degrades_to_empty(monkeypatch, get_stub):
    monkeypatch.setattr(openrouter.httpx, "get", get_stub)
    assert openrouter._free_models() == []


# ===== Helper Functions =====


@pytest.mark.parametrize(
    "exc, expected",
    [
        pytest.param(_StatusErr("boom", status_code=429), True, id="status_code_429"),
        pytest.param(ValueError("Rate limit exceeded, try later"), True, id="message_substring"),
        pytest.param(ValueError("totally unrelated"), False, id="negative"),
    ],
)
def test_is_rate_limit(exc, expected):
    assert openrouter._is_rate_limit(exc) is expected


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
    assert openrouter._is_auth(exc) is expected
