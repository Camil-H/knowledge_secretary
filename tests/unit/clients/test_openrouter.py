"""The OpenRouter tier: its model catalog, one chat completion, and the deadline-bounded loop
over the ranked candidates. httpx and sleep are faked — no real request, key or wait."""

import pytest

import src.clients.openrouter as openrouter
from src import config
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


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No sleeps, and a key present so the auth guard is not what fails."""
    monkeypatch.setattr(openrouter.time, "sleep", lambda _s: None)
    monkeypatch.setenv(config.OPENROUTER_KEY_LABEL, "or-key")


# ===== Tier =====

# ----- rate-limit backoff + fall-through -----


def test_call_retries_the_same_model_on_rate_limit(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MODELS", ["openrouter/a:free"])
    n = {"i": 0}

    def _complete(model, messages, max_tokens):
        n["i"] += 1
        if n["i"] == 1:
            raise _StatusErr("rate limit exceeded", status_code=429)
        return "ok"

    monkeypatch.setattr(openrouter, "complete", _complete)
    assert openrouter.call("s", "u", None) == "ok"
    assert n["i"] == 2


@pytest.mark.parametrize(
    "error",
    [ValueError("boom"), _StatusErr("rate limit reached", status_code=402)],
    ids=["untyped", "non_transient_status_naming_a_rate_limit"],
)
def test_call_falls_through_to_the_next_model_on_other_error(monkeypatch, error):
    """A status outside TRANSIENT_STATUSES moves on rather than retrying, whatever its body says."""
    monkeypatch.setattr(config, "OPENROUTER_MODELS", ["openrouter/a:free", "openrouter/b:free"])
    tried = []

    def _complete(model, messages, max_tokens):
        tried.append(model)
        if model == "openrouter/a:free":
            raise error
        return "second"

    monkeypatch.setattr(openrouter, "complete", _complete)
    assert openrouter.call("s", "u", None) == "second"
    assert tried == ["openrouter/a:free", "openrouter/b:free"]


def test_call_raises_auth_error_immediately(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MODELS", ["openrouter/a:free", "openrouter/b:free"])
    tried = []

    def _complete(model, messages, max_tokens):
        tried.append(model)
        raise _StatusErr("unauthorized", status_code=401)

    monkeypatch.setattr(openrouter, "complete", _complete)
    with pytest.raises(AuthError):
        openrouter.call("s", "u", None)
    assert tried == ["openrouter/a:free"]


def test_call_does_not_read_auth_intent_from_the_message(monkeypatch):
    """A 401 is a 401; an untyped error whose text mentions credentials is just a failed
    candidate, where the old phrase list would have raised AuthError and stopped the cascade."""
    monkeypatch.setattr(config, "OPENROUTER_MODELS", ["openrouter/a:free", "openrouter/b:free"])

    def _complete(model, messages, max_tokens):
        if model == "openrouter/a:free":
            raise ValueError("No user or org id found in auth cookie")
        return "second"

    monkeypatch.setattr(openrouter, "complete", _complete)
    assert openrouter.call("s", "u", None) == "second"


def test_call_external_error_carries_the_last_exception_as_cause(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MODELS", ["openrouter/a:free"])
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
    monkeypatch.setattr(config, "OPENROUTER_MODELS", candidates)
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

    assert calls["a"] == config.RATE_LIMIT_RETRIES


def test_call_backoff_doubles_and_caps(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MODELS", ["openrouter/a:free"])
    retries = 7
    monkeypatch.setattr(config, "RATE_LIMIT_RETRIES", retries)
    sleeps: list[float] = []
    monkeypatch.setattr(openrouter.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(openrouter, "complete", _raiser(_StatusErr("rate limit", status_code=429)))

    with pytest.raises(ExternalError):
        openrouter.call("s", "u", None)

    expected = []
    backoff = config.BACKOFF_START_S
    for _ in range(retries - 1):
        expected.append(backoff)
        backoff = min(backoff * 2, config.BACKOFF_CAP_S)
    assert sleeps == expected


# ----- deadline / budget -----


def test_call_abandons_the_cascade_once_the_deadline_is_exceeded(monkeypatch):
    _fake_clock(monkeypatch)
    monkeypatch.setattr(config, "OPENROUTER_DEADLINE_S", 5.0)
    candidates = [f"openrouter/m{i}:free" for i in range(4)]
    monkeypatch.setattr(config, "OPENROUTER_MODELS", candidates)
    tried = []

    def _complete(model, messages, max_tokens):
        tried.append(model)
        raise _StatusErr("rate limit exceeded", status_code=429)

    monkeypatch.setattr(openrouter, "complete", _complete)
    with pytest.raises(ExternalError, match="all models failed"):
        openrouter.call("s", "u", None)

    assert len(set(tried)) < len(candidates)
    assert len(tried) < len(candidates) * config.RATE_LIMIT_RETRIES


# ----- empty / whitespace content -----


@pytest.mark.parametrize("empty_content", ["", "   ", "\n\t "])
def test_call_falls_through_on_empty_or_whitespace_content(monkeypatch, empty_content):
    monkeypatch.setattr(config, "OPENROUTER_MODELS", ["openrouter/a:free", "openrouter/b:free"])

    def _complete(model, messages, max_tokens):
        return empty_content if model == "openrouter/a:free" else "second"

    monkeypatch.setattr(openrouter, "complete", _complete)
    assert openrouter.call("s", "u", None) == "second"


def test_call_raises_when_all_models_return_empty(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MODELS", ["openrouter/a:free", "openrouter/b:free"])
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
    assert seen["timeout"] == config.HTTP_TIMEOUT_S


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


@pytest.mark.parametrize("status", [429, 401, 402, 503], ids=str)
def test_complete_raises_an_error_carrying_the_status(monkeypatch, status):
    """The tier classifies on status_code alone, so surfacing it is complete()'s contract."""
    _capture_post(monkeypatch, _FakeResp(None, status_code=status, text='{"error": "rate limit"}'))

    with pytest.raises(openrouter.OpenRouterError) as ei:
        openrouter.complete("openrouter/a:free", [], None)
    assert ei.value.status_code == status


def test_complete_truncates_the_error_body(monkeypatch):
    _capture_post(monkeypatch, _FakeResp(None, status_code=500, text="x" * 5000))

    with pytest.raises(openrouter.OpenRouterError) as ei:
        openrouter.complete("openrouter/a:free", [], None)
    assert len(str(ei.value)) < openrouter._MAX_ERROR_BODY_CHARS * 2


def test_complete_raises_auth_error_without_a_key(monkeypatch):
    monkeypatch.delenv(config.OPENROUTER_KEY_LABEL, raising=False)
    monkeypatch.setattr(openrouter.httpx, "post", _raiser(AssertionError("must not be called")))

    with pytest.raises(AuthError, match=config.OPENROUTER_KEY_LABEL):
        openrouter.complete("openrouter/a:free", [], None)
