"""Google AI Studio tier, OpenRouter tier, and the cascade between them. The genai client,
httpx, the ledger path (via chdir) and sleep are all faked — no real request, key or wait."""

import httpx
import pytest
from google.genai import errors as genai_errors

import src.core.llm as llm
from src.core import ledger as ledger_mod
from src.core.errors import AuthError, ExternalError, QuotaExhausted

# ----- test doubles -----

_MINUTE_QUOTA = {
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "details": [{"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}]}],
    }
}
_DAY_QUOTA = {
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}],
    }
}
_UNSCOPED_QUOTA = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}


def _api_error(code: int, payload: dict | None = None) -> genai_errors.APIError:
    return genai_errors.ClientError(code, payload or {"error": {"code": code}})


class _FakeGeminiResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeGenaiModels:
    """Stands in for client.models: replays a scripted sequence, recording every call."""

    def __init__(self, script) -> None:
        self._script = list(script)
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self._script[min(len(self.calls) - 1, len(self._script) - 1)]
        if isinstance(item, dict):
            item = item.get(model, _FakeGeminiResponse("ok"))
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def models_tried(self) -> list[str]:
        return [c["model"] for c in self.calls]


class _FakeGenaiClient:
    def __init__(self, script) -> None:
        self.models = _FakeGenaiModels(script)


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
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(llm.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    return clock


def _fake_google(monkeypatch, script) -> _FakeGenaiModels:
    client = _FakeGenaiClient(script)
    monkeypatch.setattr(llm, "_CLIENT", client)
    return client.models


def _install_ledger(monkeypatch, ledger: dict) -> None:
    monkeypatch.setattr(llm.ledger_mod, "load", lambda *_a, **_k: ledger)


def _openrouter_returning(text: str):
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
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: _FakeResp(_MODELS))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No sleeps, no memoized client, no pacing carried between tests, and the ledger's
    write-through confined to tmp_path via cwd (its path default binds at def time)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    monkeypatch.setattr(llm, "_CLIENT", None)
    monkeypatch.setattr(llm, "_NEXT_DISPATCH", {})
    monkeypatch.delenv(llm.GOOGLE_KEY_LABEL, raising=False)
    monkeypatch.setenv(llm.OPENROUTER_KEY_LABEL, "or-key")
    llm._reset_model_cache()
    yield
    llm._reset_model_cache()


# ===== Google AI Studio primitive =====

# ----- dispatch + composition -----


def _config(max_output_tokens: int | None = None):
    return llm.types.GenerateContentConfig(
        system_instruction="be brief", max_output_tokens=max_output_tokens
    )


def test_gemini_generate_returns_the_sdk_response(monkeypatch):
    response = _FakeGeminiResponse("an answer")
    _fake_google(monkeypatch, [response])
    model = llm.GEMINI_TEXT_MODELS[0]

    assert (
        llm.gemini_generate(model, "hi", _config(), ledger=ledger_mod.load()) is response
    )  # the raw response, so callers can read grounding metadata


def test_gemini_generate_sends_the_model_contents_and_config(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    model = llm.GEMINI_TEXT_MODELS[1]
    config = _config(max_output_tokens=321)

    llm.gemini_generate(model, "the prompt", config, ledger=ledger_mod.load())

    assert models.calls == [{"model": model.id, "contents": "the prompt", "config": config}]


def test_gemini_generate_raises_auth_error_without_a_key(monkeypatch):
    calls = []
    monkeypatch.setattr(llm.genai, "Client", lambda **kw: calls.append(kw))
    ledger = ledger_mod.load()

    with pytest.raises(AuthError, match=llm.GOOGLE_KEY_LABEL):
        llm.gemini_generate(llm.GEMINI_TEXT_MODELS[0], "hi", _config(), ledger=ledger)

    assert calls == []
    assert ledger["models"] == {}  # nothing was dispatched, so nothing was spent


@pytest.mark.parametrize(
    "error, expected",
    [
        pytest.param(_api_error(401), AuthError, id="unauthorized"),
        pytest.param(_api_error(403), AuthError, id="forbidden"),
        pytest.param(_api_error(500), ExternalError, id="server_error"),
        pytest.param(RuntimeError("socket died"), ExternalError, id="non_api_error"),
    ],
)
def test_gemini_generate_types_sdk_failures(monkeypatch, error, expected):
    _fake_google(monkeypatch, [error])

    with pytest.raises(expected) as ei:
        llm.gemini_generate(llm.GEMINI_TEXT_MODELS[0], "hi", _config(), ledger=ledger_mod.load())

    assert ei.value.cause is error


# ----- ledger accounting -----


@pytest.mark.parametrize(
    "script, expected_dispatches",
    [
        pytest.param([_FakeGeminiResponse("ok")], 1, id="success"),
        pytest.param([_api_error(429, _DAY_QUOTA)], 1, id="day_quota_429"),
        pytest.param([_api_error(500)], 1, id="server_error"),
    ],
)
def test_gemini_generate_consumes_budget_at_dispatch(monkeypatch, script, expected_dispatches):
    """A dispatched request is spent even when it fails — the provider may have counted it."""
    models = _fake_google(monkeypatch, script)
    model = llm.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    try:
        llm.gemini_generate(model, "hi", _config(), ledger=ledger)
    except ExternalError:
        pass

    assert ledger["models"][model.id]["requests"] == expected_dispatches
    assert len(models.calls) == expected_dispatches


def test_gemini_generate_writes_the_dispatch_through_to_disk(monkeypatch):
    _fake_google(monkeypatch, [_api_error(429, _DAY_QUOTA)])
    model = llm.GEMINI_TEXT_MODELS[0]

    with pytest.raises(QuotaExhausted):
        llm.gemini_generate(model, "hi", _config(), ledger=ledger_mod.load())

    reloaded = ledger_mod.load()
    assert reloaded["models"][model.id]["requests"] == 1
    assert not ledger_mod.available(reloaded, model.id, model.rpd)


# ----- 429 handling -----


def test_gemini_generate_day_quota_retires_the_model_immediately(monkeypatch):
    models = _fake_google(monkeypatch, [_api_error(429, _DAY_QUOTA)])
    model = llm.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    with pytest.raises(QuotaExhausted, match=model.id):
        llm.gemini_generate(model, "hi", _config(), ledger=ledger)

    assert len(models.calls) == 1  # a day limit is not worth retrying
    assert ledger["models"][model.id]["exhausted"] is True


@pytest.mark.parametrize("payload", [_MINUTE_QUOTA, _UNSCOPED_QUOTA], ids=["minute", "unscoped"])
def test_gemini_generate_retries_a_transient_429_then_retires_the_model(monkeypatch, payload):
    models = _fake_google(monkeypatch, [_api_error(429, payload)])
    model = llm.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    with pytest.raises(QuotaExhausted):
        llm.gemini_generate(model, "hi", _config(), ledger=ledger)

    assert len(models.calls) == llm._RATE_LIMIT_RETRIES
    assert ledger["models"][model.id]["requests"] == llm._RATE_LIMIT_RETRIES


def test_gemini_generate_succeeds_after_a_minute_quota_retry(monkeypatch):
    models = _fake_google(
        monkeypatch, [_api_error(429, _MINUTE_QUOTA), _FakeGeminiResponse("second try")]
    )
    model = llm.GEMINI_TEXT_MODELS[0]

    response = llm.gemini_generate(model, "hi", _config(), ledger=ledger_mod.load())

    assert response.text == "second try"
    assert len(models.calls) == 2


def test_gemini_generate_backoff_doubles_and_caps(monkeypatch):
    retries = 7  # enough attempts for the doubling sequence to actually hit the cap
    monkeypatch.setattr(llm, "_RATE_LIMIT_RETRIES", retries)
    _fake_google(monkeypatch, [_api_error(429, _MINUTE_QUOTA)])
    monkeypatch.setattr(llm, "_pace", lambda *_a: None)  # pacing sleeps have their own tests
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(QuotaExhausted):
        llm.gemini_generate(llm.GEMINI_TEXT_MODELS[0], "hi", _config(), ledger=ledger_mod.load())

    expected = []
    backoff = llm._BACKOFF_START_S
    for _ in range(retries - 1):  # one sleep per retried attempt, none after the last
        expected.append(backoff)
        backoff = min(backoff * 2, llm._BACKOFF_CAP_S)
    assert sleeps == expected


# ----- proactive pacing -----


def test_gemini_generate_paces_consecutive_calls_to_the_same_model(monkeypatch):
    clock = _fake_clock(monkeypatch)
    _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    model = llm.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    llm.gemini_generate(model, "hi", _config(), ledger=ledger)
    llm.gemini_generate(model, "hi", _config(), ledger=ledger)

    assert clock["t"] >= 60 / model.rpm


def test_gemini_generate_paces_on_token_volume_when_it_dominates(monkeypatch):
    clock = _fake_clock(monkeypatch)
    _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    model = llm.GEMINI_TEXT_MODELS[0]
    contents = "x" * (model.tpm * 4)  # a full minute of tokens by the len//4 estimate
    ledger = ledger_mod.load()

    llm.gemini_generate(model, contents, _config(), ledger=ledger)
    llm.gemini_generate(model, contents, _config(), ledger=ledger)

    assert clock["t"] > 60 / model.rpm


def test_gemini_generate_does_not_pace_across_different_models(monkeypatch):
    clock = _fake_clock(monkeypatch)
    _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    ledger = ledger_mod.load()

    for model in llm.GEMINI_TEXT_MODELS:
        llm.gemini_generate(model, "hi", _config(), ledger=ledger)

    assert clock["t"] == 0  # each model has its own rpm window


# ----- _quota_scope -----


@pytest.mark.parametrize(
    "payload, expected",
    [
        pytest.param(_MINUTE_QUOTA, "PerMinute", id="minute"),
        pytest.param(_DAY_QUOTA, "PerDay", id="day"),
        pytest.param(_UNSCOPED_QUOTA, None, id="unstated"),
    ],
)
def test_quota_scope(payload, expected):
    assert llm._quota_scope(_api_error(429, payload)) == expected


# ===== Cascade =====


def test_call_prefers_the_gemini_tier(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("from gemini")])
    monkeypatch.setenv(llm.GOOGLE_KEY_LABEL, "ai-studio-key")
    monkeypatch.setattr(llm, "_openrouter_complete", _openrouter_returning("from openrouter"))

    assert llm.call("s", "u") == "from gemini"
    assert models.models_tried == [llm.GEMINI_TEXT_MODELS[0].id]


def test_call_hands_the_system_prompt_and_max_tokens_to_gemini(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    monkeypatch.setenv(llm.GOOGLE_KEY_LABEL, "ai-studio-key")

    llm.call("the system prompt", "the user text", max_tokens=1234)

    call = models.calls[0]
    assert call["contents"] == "the user text"
    assert call["config"].system_instruction == "the system prompt"
    assert call["config"].max_output_tokens == 1234


def test_call_skips_models_the_ledger_has_retired(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    monkeypatch.setenv(llm.GOOGLE_KEY_LABEL, "ai-studio-key")
    first, second = llm.GEMINI_TEXT_MODELS[0], llm.GEMINI_TEXT_MODELS[1]
    ledger = ledger_mod.load()
    ledger_mod.mark_exhausted(ledger, first.id)
    _install_ledger(monkeypatch, ledger)

    assert llm.call("s", "u") == "ok"
    assert models.models_tried == [second.id]


def test_call_skips_models_that_spent_their_daily_budget(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    monkeypatch.setenv(llm.GOOGLE_KEY_LABEL, "ai-studio-key")
    first, second = llm.GEMINI_TEXT_MODELS[0], llm.GEMINI_TEXT_MODELS[1]
    ledger = ledger_mod.load()
    for _ in range(first.rpd):
        ledger_mod.consume(ledger, first.id)
    _install_ledger(monkeypatch, ledger)

    llm.call("s", "u")
    assert models.models_tried == [second.id]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(_api_error(429, _DAY_QUOTA), id="quota_exhausted"),
        pytest.param(_api_error(500), id="external_error"),
        pytest.param(_FakeGeminiResponse(""), id="empty_text"),
        pytest.param(_FakeGeminiResponse(None), id="no_text"),
    ],
)
def test_call_advances_to_the_next_gemini_model(monkeypatch, failure):
    first, second = llm.GEMINI_TEXT_MODELS[0], llm.GEMINI_TEXT_MODELS[1]
    models = _fake_google(
        monkeypatch, [{first.id: failure, second.id: _FakeGeminiResponse("second model")}]
    )
    monkeypatch.setenv(llm.GOOGLE_KEY_LABEL, "ai-studio-key")

    assert llm.call("s", "u") == "second model"
    assert models.models_tried[-1] == second.id


def test_call_degrades_to_openrouter_when_the_google_tier_auth_fails(monkeypatch, caplog):
    """The live incident: an AI Studio key bound to the wrong project 403s every model, and
    that must not take the newsletter and youtube digests down with it."""
    _fake_google(monkeypatch, [_api_error(403)])
    monkeypatch.setenv(llm.GOOGLE_KEY_LABEL, "wrong-project-key")
    monkeypatch.setattr(llm, "_openrouter_complete", _openrouter_returning("from openrouter"))

    with caplog.at_level("WARNING", logger=llm.logger.name):
        assert llm.call("s", "u") == "from openrouter"

    degradations = [r for r in caplog.records if "degrading to openrouter" in r.getMessage()]
    assert len(degradations) == 1  # one line per run, not one per model


def test_call_degrades_to_openrouter_when_the_google_key_is_unset(monkeypatch):
    monkeypatch.setattr(llm, "_openrouter_complete", _openrouter_returning("from openrouter"))
    assert llm.call("s", "u") == "from openrouter"


def test_call_falls_through_to_openrouter_when_every_gemini_model_is_spent(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    monkeypatch.setenv(llm.GOOGLE_KEY_LABEL, "ai-studio-key")
    ledger = ledger_mod.load()
    for model in llm.GEMINI_TEXT_MODELS:
        ledger_mod.mark_exhausted(ledger, model.id)
    _install_ledger(monkeypatch, ledger)
    monkeypatch.setattr(llm, "_openrouter_complete", _openrouter_returning("from openrouter"))

    assert llm.call("s", "u") == "from openrouter"
    assert models.calls == []


def test_call_propagates_an_openrouter_auth_failure(monkeypatch):
    """AuthError still fails loudly at the boundary: the last independent credential is gone."""
    _fake_google(monkeypatch, [_api_error(403)])
    monkeypatch.setenv(llm.GOOGLE_KEY_LABEL, "wrong-project-key")
    monkeypatch.setattr(llm, "_openrouter_models", lambda: ["openrouter/a:free"])
    monkeypatch.setattr(
        llm, "_openrouter_complete", _raiser(_StatusErr("unauthorized", status_code=401))
    )

    with pytest.raises(AuthError):
        llm.call("s", "u")


def test_call_raises_when_every_tier_is_dry(monkeypatch):
    monkeypatch.setattr(llm, "_openrouter_models", lambda: ["openrouter/a:free"])
    monkeypatch.setattr(llm, "_openrouter_complete", _raiser(ValueError("nope")))

    with pytest.raises(ExternalError, match="all models failed"):
        llm.call("s", "u")


# ===== OpenRouter tier =====

# ----- rate-limit backoff + fall-through -----


def test_openrouter_tier_retries_the_same_model_on_rate_limit(monkeypatch):
    monkeypatch.setattr(llm, "_openrouter_models", lambda: ["openrouter/a:free"])
    n = {"i": 0}

    def _complete(model, messages, max_tokens):
        n["i"] += 1
        if n["i"] == 1:
            raise _StatusErr("rate limit exceeded", status_code=429)
        return "ok"

    monkeypatch.setattr(llm, "_openrouter_complete", _complete)
    assert llm._openrouter_tier("s", "u", None) == "ok"
    assert n["i"] == 2


def test_openrouter_tier_falls_through_to_the_next_model_on_other_error(monkeypatch):
    monkeypatch.setattr(
        llm, "_openrouter_models", lambda: ["openrouter/a:free", "openrouter/b:free"]
    )

    def _complete(model, messages, max_tokens):
        if model == "openrouter/a:free":
            raise ValueError("boom")
        return "second"

    monkeypatch.setattr(llm, "_openrouter_complete", _complete)
    assert llm._openrouter_tier("s", "u", None) == "second"


@pytest.mark.parametrize("models", [[], None], ids=["empty_list", "degraded_fetch"])
def test_openrouter_tier_uses_the_fallback_model_when_none_resolve(monkeypatch, models):
    if models is None:
        monkeypatch.setattr(llm.httpx, "get", _raiser(httpx.HTTPError("boom")))
    else:
        monkeypatch.setattr(llm, "_openrouter_models", lambda: models)
    seen = {}

    def _complete(model, messages, max_tokens):
        seen["model"] = model
        return "ok"

    monkeypatch.setattr(llm, "_openrouter_complete", _complete)
    assert llm._openrouter_tier("s", "u", None) == "ok"
    assert seen["model"] == llm.FALLBACK_MODEL


def test_openrouter_tier_raises_auth_error_immediately(monkeypatch):
    monkeypatch.setattr(
        llm, "_openrouter_models", lambda: ["openrouter/a:free", "openrouter/b:free"]
    )
    tried = []

    def _complete(model, messages, max_tokens):
        tried.append(model)
        raise ValueError("No user or org id found in auth cookie")

    monkeypatch.setattr(llm, "_openrouter_complete", _complete)
    with pytest.raises(AuthError):
        llm._openrouter_tier("s", "u", None)
    assert tried == ["openrouter/a:free"]  # auth fails loudly on the first model, no fallback


def test_openrouter_tier_external_error_carries_the_last_exception_as_cause(monkeypatch):
    monkeypatch.setattr(llm, "_openrouter_models", lambda: ["openrouter/a:free"])
    boom = ValueError("nope")
    monkeypatch.setattr(llm, "_openrouter_complete", _raiser(boom))

    with pytest.raises(ExternalError) as ei:
        llm._openrouter_tier("s", "u", None)
    assert ei.value.cause is boom


@pytest.mark.parametrize(
    "models, expect_next_model",
    [
        pytest.param(["openrouter/a:free"], False, id="single_model_raises"),
        pytest.param(["openrouter/a:free", "openrouter/b:free"], True, id="advances_to_next_model"),
    ],
)
def test_openrouter_tier_persistent_rate_limit_exhausts_retries(
    monkeypatch, models, expect_next_model
):
    monkeypatch.setattr(llm, "_openrouter_models", lambda: models)
    calls = {"a": 0}

    def _complete(model, messages, max_tokens):
        if model == "openrouter/a:free":
            calls["a"] += 1
            raise _StatusErr("rate limit exceeded", status_code=429)
        return "ok"

    monkeypatch.setattr(llm, "_openrouter_complete", _complete)

    if expect_next_model:
        assert llm._openrouter_tier("s", "u", None) == "ok"
    else:
        with pytest.raises(ExternalError):
            llm._openrouter_tier("s", "u", None)

    assert calls["a"] == llm._RATE_LIMIT_RETRIES


def test_openrouter_tier_backoff_doubles_and_caps(monkeypatch):
    monkeypatch.setattr(llm, "_openrouter_models", lambda: ["openrouter/a:free"])
    retries = 7
    monkeypatch.setattr(llm, "_RATE_LIMIT_RETRIES", retries)
    sleeps: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(
        llm, "_openrouter_complete", _raiser(_StatusErr("rate limit", status_code=429))
    )

    with pytest.raises(ExternalError):
        llm._openrouter_tier("s", "u", None)

    expected = []
    backoff = llm._BACKOFF_START_S
    for _ in range(retries - 1):
        expected.append(backoff)
        backoff = min(backoff * 2, llm._BACKOFF_CAP_S)
    assert sleeps == expected


# ----- deadline / budget -----


def test_openrouter_tier_abandons_the_cascade_once_the_deadline_is_exceeded(monkeypatch):
    _fake_clock(monkeypatch)
    monkeypatch.setattr(llm, "_DEADLINE_S", 5.0)
    models = [f"openrouter/m{i}:free" for i in range(llm._FREE_LIMIT)]
    monkeypatch.setattr(llm, "_openrouter_models", lambda: models)
    tried = []

    def _complete(model, messages, max_tokens):
        tried.append(model)
        raise _StatusErr("rate limit exceeded", status_code=429)

    monkeypatch.setattr(llm, "_openrouter_complete", _complete)
    with pytest.raises(ExternalError, match="all models failed"):
        llm._openrouter_tier("s", "u", None)

    assert len(set(tried)) < len(models)
    assert len(tried) < len(models) * llm._RATE_LIMIT_RETRIES


# ----- empty / whitespace content -----


@pytest.mark.parametrize("empty_content", ["", "   ", "\n\t "])
def test_openrouter_tier_falls_through_on_empty_or_whitespace_content(monkeypatch, empty_content):
    monkeypatch.setattr(
        llm, "_openrouter_models", lambda: ["openrouter/a:free", "openrouter/b:free"]
    )

    def _complete(model, messages, max_tokens):
        return empty_content if model == "openrouter/a:free" else "second"

    monkeypatch.setattr(llm, "_openrouter_complete", _complete)
    assert llm._openrouter_tier("s", "u", None) == "second"


def test_openrouter_tier_raises_when_all_models_return_empty(monkeypatch):
    monkeypatch.setattr(
        llm, "_openrouter_models", lambda: ["openrouter/a:free", "openrouter/b:free"]
    )
    monkeypatch.setattr(llm, "_openrouter_complete", _openrouter_returning("   "))
    with pytest.raises(ExternalError, match="all models failed"):
        llm._openrouter_tier("s", "u", None)


# ----- _openrouter_complete -----


def _capture_post(monkeypatch, response: _FakeResp) -> dict:
    seen: dict = {}

    def _post(url, *, headers, json, timeout):
        seen.update(url=url, headers=headers, json=json, timeout=timeout)
        return response

    monkeypatch.setattr(llm.httpx, "post", _post)
    return seen


def test_openrouter_complete_posts_the_openai_shaped_body(monkeypatch):
    seen = _capture_post(monkeypatch, _FakeResp({"choices": [{"message": {"content": "hello"}}]}))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    assert llm._openrouter_complete("openrouter/vendor/model:free", messages, 99) == "hello"
    assert seen["url"] == llm.OPENROUTER_COMPLETIONS_URL
    assert seen["json"] == {
        "model": "vendor/model:free",
        "messages": messages,
        "max_tokens": 99,
    }
    assert seen["headers"]["Authorization"].endswith("or-key")
    assert seen["timeout"] == llm._HTTP_TIMEOUT_S


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
def test_openrouter_complete_returns_empty_on_a_textless_response(monkeypatch, payload):
    _capture_post(monkeypatch, _FakeResp(payload))
    assert llm._openrouter_complete("openrouter/a:free", [], None) == ""


@pytest.mark.parametrize(
    "status, body, expected",
    [
        pytest.param(429, '{"error": "rate limit"}', llm._is_rate_limit, id="rate_limited"),
        pytest.param(401, '{"error": "unauthorized"}', llm._is_auth, id="unauthorized"),
        pytest.param(402, '{"error": "rate limit reached"}', llm._is_rate_limit, id="body_only"),
    ],
)
def test_openrouter_complete_raises_a_classifiable_error(monkeypatch, status, body, expected):
    _capture_post(monkeypatch, _FakeResp(None, status_code=status, text=body))

    with pytest.raises(llm.OpenRouterError) as ei:
        llm._openrouter_complete("openrouter/a:free", [], None)
    assert expected(ei.value)


def test_openrouter_complete_truncates_the_error_body(monkeypatch):
    _capture_post(monkeypatch, _FakeResp(None, status_code=500, text="x" * 5000))

    with pytest.raises(llm.OpenRouterError) as ei:
        llm._openrouter_complete("openrouter/a:free", [], None)
    assert len(str(ei.value)) < llm._MAX_ERROR_BODY_CHARS * 2


def test_openrouter_complete_raises_auth_error_without_a_key(monkeypatch):
    monkeypatch.delenv(llm.OPENROUTER_KEY_LABEL, raising=False)
    monkeypatch.setattr(llm.httpx, "post", _raiser(AssertionError("must not be called")))

    with pytest.raises(AuthError, match=llm.OPENROUTER_KEY_LABEL):
        llm._openrouter_complete("openrouter/a:free", [], None)


# ===== Model resolution =====


def test_free_filter_excludes_paid(monkeypatch):
    _patch_models(monkeypatch)
    assert all("paid" not in m for m in llm._free_openrouter_models())


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
    assert llm._free_openrouter_models() == ["openrouter/good/writer"]


def test_free_openrouter_models_ranks_by_context(monkeypatch):
    _patch_models(monkeypatch)
    assert llm._free_openrouter_models() == ["openrouter/big-ctx", "openrouter/small-ctx"]


def test_free_openrouter_models_rank_key_handles_a_missing_context_length(monkeypatch):
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
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: _FakeResp(edge))
    assert llm._free_openrouter_models()[0] == "openrouter/has-context"


def test_openrouter_models_prefers_curated_ids_present_in_live_list(monkeypatch):
    preferred_first, preferred_second = llm.PREFERRED_CONTEXT[0], llm.PREFERRED_CONTEXT[1]
    models = {
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
    monkeypatch.setattr(llm.httpx, "get", lambda *a, **k: _FakeResp(models))

    result = llm._openrouter_models()

    assert result[:2] == [preferred_first, preferred_second]
    assert result[2] == "openrouter/big-ctx"


def test_openrouter_models_skips_absent_preferred_ids_without_crashing(monkeypatch):
    _patch_models(monkeypatch)  # none of PREFERRED_CONTEXT's ids are present
    assert llm._openrouter_models() == ["openrouter/big-ctx", "openrouter/small-ctx"]


# ----- catalog cache + degradation -----


def test_free_openrouter_models_memoizes_the_catalog(monkeypatch):
    calls = {"n": 0}

    def _get(*_a, **_k):
        calls["n"] += 1
        return _FakeResp(_MODELS)

    monkeypatch.setattr(llm.httpx, "get", _get)

    first = llm._free_openrouter_models()
    assert llm._free_openrouter_models() == first
    assert calls["n"] == 1


def test_free_openrouter_models_does_not_cache_a_failed_fetch(monkeypatch):
    calls = {"n": 0}

    def _get(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.HTTPError("boom")
        return _FakeResp(_MODELS)

    monkeypatch.setattr(llm.httpx, "get", _get)

    assert llm._free_openrouter_models() == []
    assert llm._free_openrouter_models() != []
    assert calls["n"] == 2


@pytest.mark.parametrize(
    "get_stub",
    [
        pytest.param(_raiser(httpx.HTTPError("boom")), id="http_error"),
        pytest.param(lambda *a, **k: _FakeResp({"unexpected": []}), id="missing_data_key"),
    ],
)
def test_free_openrouter_models_degrades_to_empty(monkeypatch, get_stub):
    monkeypatch.setattr(llm.httpx, "get", get_stub)
    assert llm._free_openrouter_models() == []


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
