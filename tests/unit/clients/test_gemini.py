"""The Google AI Studio primitive and the tier loop over its model table. The genai client,
the ledger path (via chdir) and sleep are all faked — no real request, key or wait."""

import logging

import pytest
from google.genai import errors as genai_errors

import src.clients.gemini as gemini
from src import config
from src.core import ledger as ledger_mod
from src.core.errors import AuthError, ExternalError, QuotaExhausted
from src.core.models import ModelLimit

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


class _FakeWeb:
    def __init__(self, uri: str) -> None:
        self.uri = uri


class _FakeChunk:
    def __init__(self, uri: str) -> None:
        self.web = _FakeWeb(uri) if uri else None


class _FakeGrounding:
    def __init__(self, uris: list[str]) -> None:
        self.grounding_chunks = [_FakeChunk(u) for u in uris]


class _FakeCandidate:
    def __init__(self, uris: list[str] | None) -> None:
        self.grounding_metadata = _FakeGrounding(uris) if uris is not None else None


class _FakeGeminiResponse:
    def __init__(self, text: str | None, uris: list[str] | None = None) -> None:
        self.text = text
        self.candidates = [_FakeCandidate(uris)]


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


def _fake_clock(monkeypatch, start: float = 0.0):
    """Deterministic monotonic clock; sleep advances it so backoff spends the budget."""
    clock = {"t": start}
    monkeypatch.setattr(gemini.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(gemini.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    return clock


def _fake_google(monkeypatch, script) -> _FakeGenaiModels:
    client = _FakeGenaiClient(script)
    monkeypatch.setattr(gemini, "_CLIENT", client)
    return client.models


def _config(max_output_tokens: int | None = None):
    return gemini.types.GenerateContentConfig(
        system_instruction="be brief", max_output_tokens=max_output_tokens
    )


def _search_models() -> list[ModelLimit]:
    return [m for m in config.GEMINI_TEXT_MODELS if m.search]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No sleeps, no memoized client, no pacing carried between tests, and the ledger's
    write-through confined to tmp_path via cwd (its path default binds at def time)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gemini.time, "sleep", lambda _s: None)
    monkeypatch.setattr(gemini, "_CLIENT", None)
    monkeypatch.setattr(gemini, "_NEXT_DISPATCH", {})
    monkeypatch.delenv(config.GEMINI_KEY_LABEL, raising=False)


# ===== Primitive =====

# ----- dispatch + composition -----


def test_generate_returns_the_sdk_response(monkeypatch):
    response = _FakeGeminiResponse("an answer")
    _fake_google(monkeypatch, [response])
    model = config.GEMINI_TEXT_MODELS[0]

    assert gemini.generate(model, "hi", _config(), ledger=ledger_mod.load()) is response


def test_generate_sends_the_model_contents_and_config(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    model = config.GEMINI_TEXT_MODELS[1]
    gen_config = _config(max_output_tokens=321)

    gemini.generate(model, "the prompt", gen_config, ledger=ledger_mod.load())

    assert models.calls == [{"model": model.id, "contents": "the prompt", "config": gen_config}]


def test_generate_raises_auth_error_without_a_key(monkeypatch):
    calls = []
    monkeypatch.setattr(gemini.genai, "Client", lambda **kw: calls.append(kw))
    ledger = ledger_mod.load()

    with pytest.raises(AuthError, match=config.GEMINI_KEY_LABEL):
        gemini.generate(config.GEMINI_TEXT_MODELS[0], "hi", _config(), ledger=ledger)

    assert calls == []
    assert ledger == {}


@pytest.mark.parametrize(
    "error, expected",
    [
        pytest.param(_api_error(401), AuthError, id="unauthorized"),
        pytest.param(_api_error(403), AuthError, id="forbidden"),
        pytest.param(_api_error(500), ExternalError, id="server_error"),
        pytest.param(RuntimeError("socket died"), ExternalError, id="non_api_error"),
    ],
)
def test_generate_types_sdk_failures(monkeypatch, error, expected):
    _fake_google(monkeypatch, [error])

    with pytest.raises(expected) as ei:
        gemini.generate(config.GEMINI_TEXT_MODELS[0], "hi", _config(), ledger=ledger_mod.load())

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
def test_generate_consumes_budget_at_dispatch(monkeypatch, script, expected_dispatches):
    """A dispatched request is spent even when it fails — the provider may have counted it."""
    models = _fake_google(monkeypatch, script)
    model = config.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    try:
        gemini.generate(model, "hi", _config(), ledger=ledger)
    except ExternalError:
        pass

    assert ledger[model.id]["requests"] == expected_dispatches
    assert len(models.calls) == expected_dispatches


def test_generate_writes_the_dispatch_through_to_disk(monkeypatch):
    _fake_google(monkeypatch, [_api_error(429, _DAY_QUOTA)])
    model = config.GEMINI_TEXT_MODELS[0]

    with pytest.raises(QuotaExhausted):
        gemini.generate(model, "hi", _config(), ledger=ledger_mod.load())

    reloaded = ledger_mod.load()
    assert reloaded[model.id]["requests"] == 1
    assert not ledger_mod.available(reloaded, model.id, model.rpd)


# ----- 429 handling -----


def test_generate_day_quota_retires_the_model_immediately(monkeypatch):
    models = _fake_google(monkeypatch, [_api_error(429, _DAY_QUOTA)])
    model = config.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    with pytest.raises(QuotaExhausted, match=model.id):
        gemini.generate(model, "hi", _config(), ledger=ledger)

    assert len(models.calls) == 1
    assert ledger[model.id]["exhausted"] is True


def test_generate_retires_a_model_the_api_does_not_know(monkeypatch):
    """An unknown model id is permanent: without retiring it, every later call in the day pays
    its pacing delay and a failed request again. No id in the table is verified against a live
    API, so this is the expected fate of a wrong one."""
    models = _fake_google(monkeypatch, [_api_error(404), _FakeGeminiResponse("ok")])
    model = config.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    with pytest.raises(ExternalError):
        gemini.generate(model, "hi", _config(), ledger=ledger)

    assert len(models.calls) == 1
    assert not ledger_mod.available(ledger, model.id, model.rpd)


@pytest.mark.parametrize("payload", [_MINUTE_QUOTA, _UNSCOPED_QUOTA], ids=["minute", "unscoped"])
def test_generate_retries_a_transient_429_then_retires_the_model(monkeypatch, payload):
    models = _fake_google(monkeypatch, [_api_error(429, payload)])
    model = config.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    with pytest.raises(QuotaExhausted):
        gemini.generate(model, "hi", _config(), ledger=ledger)

    assert len(models.calls) == config.RATE_LIMIT_RETRIES
    assert ledger[model.id]["requests"] == config.RATE_LIMIT_RETRIES


def test_generate_succeeds_after_a_minute_quota_retry(monkeypatch):
    models = _fake_google(
        monkeypatch, [_api_error(429, _MINUTE_QUOTA), _FakeGeminiResponse("second try")]
    )
    model = config.GEMINI_TEXT_MODELS[0]

    response = gemini.generate(model, "hi", _config(), ledger=ledger_mod.load())

    assert response.text == "second try"
    assert len(models.calls) == 2


def test_generate_backoff_doubles_and_caps(monkeypatch):
    retries = 7  # enough attempts for the doubling sequence to actually hit the cap
    monkeypatch.setattr(config, "RATE_LIMIT_RETRIES", retries)
    _fake_google(monkeypatch, [_api_error(429, _MINUTE_QUOTA)])
    monkeypatch.setattr(gemini, "_pace", lambda *_a: None)  # pacing sleeps have their own tests
    sleeps: list[float] = []
    monkeypatch.setattr(gemini.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(QuotaExhausted):
        gemini.generate(config.GEMINI_TEXT_MODELS[0], "hi", _config(), ledger=ledger_mod.load())

    expected = []
    backoff = config.BACKOFF_START_S
    for _ in range(retries - 1):  # one sleep per retried attempt, none after the last
        expected.append(backoff)
        backoff = min(backoff * 2, config.BACKOFF_CAP_S)
    assert sleeps == expected


# ----- proactive pacing -----


@pytest.mark.parametrize("model", config.GEMINI_TEXT_MODELS, ids=lambda m: m.id)
def test_generate_paces_consecutive_calls_to_the_same_model(monkeypatch, model):
    """Spacing comes from the row's own rpm, so a model with a wider allowance waits less."""
    clock = _fake_clock(monkeypatch)
    _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    ledger = ledger_mod.load()

    gemini.generate(model, "hi", _config(), ledger=ledger)
    gemini.generate(model, "hi", _config(), ledger=ledger)

    assert clock["t"] == pytest.approx(60 / model.rpm)


def test_generate_paces_on_token_volume_when_it_dominates(monkeypatch):
    clock = _fake_clock(monkeypatch)
    _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    model = config.GEMINI_TEXT_MODELS[0]
    contents = "x" * (model.tpm * 4)  # a full minute of tokens by the len//4 estimate
    ledger = ledger_mod.load()

    gemini.generate(model, contents, _config(), ledger=ledger)
    gemini.generate(model, contents, _config(), ledger=ledger)

    assert clock["t"] > 60 / model.rpm


def test_generate_does_not_pace_across_different_models(monkeypatch):
    clock = _fake_clock(monkeypatch)
    _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    ledger = ledger_mod.load()

    for model in config.GEMINI_TEXT_MODELS:
        gemini.generate(model, "hi", _config(), ledger=ledger)

    assert clock["t"] == 0


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
    assert gemini._quota_scope(_api_error(429, payload)) == expected


# ===== Tier =====

# ----- call -----


def test_call_hands_the_system_prompt_and_max_tokens_to_the_model(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])

    gemini.call("the system prompt", "the user text", 1234, ledger=ledger_mod.load())

    call = models.calls[0]
    assert call["contents"] == "the user text"
    assert call["config"].system_instruction == "the system prompt"
    assert call["config"].max_output_tokens == 1234


def test_call_attaches_no_tool(monkeypatch):
    """Grounded search belongs to podcast research alone. This is the newsletter, youtube and
    transcript path, and a tool leaking into it would put a billed web search behind every item
    of every run without changing any visible output."""
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])

    gemini.call("s", "u", None, ledger=ledger_mod.load())

    assert models.calls[0]["config"].tools is None


def test_call_skips_models_the_ledger_has_retired(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    first, second = config.GEMINI_TEXT_MODELS[0], config.GEMINI_TEXT_MODELS[1]
    ledger = ledger_mod.load()
    ledger_mod.mark_exhausted(ledger, first.id)

    assert gemini.call("s", "u", None, ledger=ledger) == "ok"
    assert models.models_tried == [second.id]


def test_call_skips_models_that_spent_their_daily_budget(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    first, second = config.GEMINI_TEXT_MODELS[0], config.GEMINI_TEXT_MODELS[1]
    ledger = ledger_mod.load()
    for _ in range(first.rpd):
        ledger_mod.consume(ledger, first.id)

    gemini.call("s", "u", None, ledger=ledger)
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
def test_call_advances_to_the_next_model(monkeypatch, failure):
    first, second = config.GEMINI_TEXT_MODELS[0], config.GEMINI_TEXT_MODELS[1]
    models = _fake_google(
        monkeypatch, [{first.id: failure, second.id: _FakeGeminiResponse("second model")}]
    )

    assert gemini.call("s", "u", None, ledger=ledger_mod.load()) == "second model"
    assert models.models_tried[-1] == second.id


def test_call_propagates_an_auth_error_instead_of_advancing(monkeypatch):
    """A credential failure fails every model identically, so advancing would spend four models'
    pacing and retries per call before giving up. Raising is what lets llm.call degrade to the
    OpenRouter tier, which is the incident this cascade exists to survive."""
    first, second = config.GEMINI_TEXT_MODELS[0], config.GEMINI_TEXT_MODELS[1]
    models = _fake_google(
        monkeypatch,
        [{first.id: _api_error(401), second.id: _FakeGeminiResponse("must not be reached")}],
    )

    with pytest.raises(AuthError):
        gemini.call("s", "u", None, ledger=ledger_mod.load())
    assert models.models_tried == [first.id]


def test_call_returns_empty_when_every_model_is_spent(monkeypatch):
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    ledger = ledger_mod.load()
    for model in config.GEMINI_TEXT_MODELS:
        ledger_mod.mark_exhausted(ledger, model.id)

    assert gemini.call("s", "u", None, ledger=ledger) == ""
    assert models.calls == []


@pytest.mark.parametrize(
    "search, mode",
    [pytest.param(False, "plain", id="plain"), pytest.param(True, "grounded", id="grounded")],
)
@pytest.mark.parametrize(
    "failure, expected",
    [
        pytest.param(_api_error(500), "unavailable", id="external_error"),
        pytest.param(_api_error(429, _DAY_QUOTA), "unavailable", id="quota_exhausted"),
        pytest.param(_FakeGeminiResponse(""), "returned empty", id="empty_text"),
    ],
)
def test_call_names_the_mode_in_its_warnings(monkeypatch, caplog, search, mode, failure, expected):
    """Both modes warn from this module, so the mode is what tells a grounded line from a plain
    one."""
    model = (_search_models() if search else config.GEMINI_TEXT_MODELS)[0]
    _fake_google(monkeypatch, [failure])

    with caplog.at_level(logging.WARNING, logger=gemini.logger.name):
        gemini.call("s", "u", ledger=ledger_mod.load(), search=search)

    assert [
        r.getMessage()
        for r in caplog.records
        if f"{mode} model={model.id}" in r.getMessage() and expected in r.getMessage()
    ]


# ----- grounded search -----


def test_call_with_search_attaches_the_google_search_tool(monkeypatch):
    """Without the tool the answer is ungrounded, which is the whole point of research."""
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])

    gemini.call("s", "u", ledger=ledger_mod.load(), search=True)

    assert [t for t in models.calls[0]["config"].tools if t.google_search is not None]


def test_call_with_search_tries_only_grounding_capable_models(monkeypatch):
    """The roster carries a high-quota model without grounding; handing it the search tool would
    fail the request, so a grounded call must never reach it."""
    plain = ModelLimit("no-search", rpd=500, rpm=15, tpm=250_000, search=False)
    grounded = ModelLimit("with-search", rpd=20, rpm=5, tpm=250_000, search=True)
    monkeypatch.setattr(config, "GEMINI_TEXT_MODELS", [plain, grounded])
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])

    gemini.call("s", "u", ledger=ledger_mod.load(), search=True)

    assert models.models_tried == [grounded.id]


@pytest.mark.parametrize(
    "uris, expected_in_log",
    [
        pytest.param(["https://a.example", "https://b.example"], "https://a.example", id="uris"),
        pytest.param([], None, id="no_chunks"),
        pytest.param(None, None, id="no_metadata"),
    ],
)
def test_call_with_search_logs_the_grounded_sources(monkeypatch, caplog, uris, expected_in_log):
    """The grounded URIs are the only record of what an episode was built from."""
    _fake_google(monkeypatch, [_FakeGeminiResponse("an overview", uris=uris)])

    with caplog.at_level(logging.INFO, logger=gemini.logger.name):
        gemini.call("s", "u", ledger=ledger_mod.load(), search=True)

    grounded = [r.getMessage() for r in caplog.records if "grounded in" in r.getMessage()]
    if expected_in_log:
        assert any(expected_in_log in message for message in grounded)
    else:
        assert grounded == []


def test_call_with_search_logs_the_sources_of_a_response_it_skips_as_empty(monkeypatch, caplog):
    """An empty answer still ran a search; its sources are part of the record either way."""
    first, second = _search_models()[:2]
    empty = _FakeGeminiResponse("", uris=["https://skipped.example"])
    _fake_google(monkeypatch, [{first.id: empty, second.id: _FakeGeminiResponse("an answer")}])

    with caplog.at_level(logging.INFO, logger=gemini.logger.name):
        assert gemini.call("s", "u", ledger=ledger_mod.load(), search=True) == "an answer"

    assert any("https://skipped.example" in r.getMessage() for r in caplog.records)


def test_call_without_search_logs_no_sources(monkeypatch, caplog):
    _fake_google(monkeypatch, [_FakeGeminiResponse("ok", uris=["https://a.example"])])

    with caplog.at_level(logging.INFO, logger=gemini.logger.name):
        gemini.call("s", "u", ledger=ledger_mod.load())

    assert not [r for r in caplog.records if "grounded" in r.getMessage()]
