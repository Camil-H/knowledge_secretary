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
_FREE_GROUNDING_FAMILY = "gemini-2.5-"


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


def _one_model_per_rpm() -> list[ModelLimit]:
    """First row of each distinct rpm: the pacing gap is 60/rpm, so equal-rpm rows would repeat
    the same assertion, and a new rpm in the table grows a row on its own."""
    by_rpm: dict[int, ModelLimit] = {}
    for model in config.GEMINI_TEXT_MODELS:
        by_rpm.setdefault(model.rpm, model)
    return list(by_rpm.values())


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


# ----- 429 handling -----


def test_generate_day_quota_retires_the_model_immediately(monkeypatch):
    """The retirement is written through on the module's default path, so the next run of the day
    reloads it instead of paying the quota again."""
    models = _fake_google(monkeypatch, [_api_error(429, _DAY_QUOTA)])
    model = config.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    with pytest.raises(QuotaExhausted, match=model.id):
        gemini.generate(model, "hi", _config(), ledger=ledger)

    assert len(models.calls) == 1
    assert ledger[model.id]["exhausted"] is True
    reloaded = ledger_mod.load()
    assert reloaded[model.id]["requests"] == 1
    assert not ledger_mod.available(reloaded, model.id, model.rpd)


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
def test_generate_retries_a_429_that_names_no_daily_quota_without_retiring(monkeypatch, payload):
    """A 429 the API does not attribute to a spent daily quota — a burst, or an account-level
    block like depleted prepay credits — must cost one run, not the rest of the day. Retiring on
    it writes a verdict into the committed ledger that outlives the credential that caused it,
    so a key fixed at noon still cannot be used until UTC midnight."""
    models = _fake_google(monkeypatch, [_api_error(429, payload)])
    model = config.GEMINI_TEXT_MODELS[0]
    ledger = ledger_mod.load()

    with pytest.raises(ExternalError) as ei:
        gemini.generate(model, "hi", _config(), ledger=ledger)

    assert not isinstance(ei.value, QuotaExhausted)
    assert len(models.calls) == config.RATE_LIMIT_RETRIES
    assert ledger[model.id]["requests"] == config.RATE_LIMIT_RETRIES
    assert ledger_mod.available(ledger, model.id, model.rpd)


def test_generate_succeeds_after_a_minute_quota_retry(monkeypatch):
    """The retry returns the raw response object, not its text, so callers can still read
    grounding metadata off a completion that needed a second attempt."""
    second = _FakeGeminiResponse("second try")
    models = _fake_google(monkeypatch, [_api_error(429, _MINUTE_QUOTA), second])
    model = config.GEMINI_TEXT_MODELS[0]

    assert gemini.generate(model, "hi", _config(), ledger=ledger_mod.load()) is second
    assert len(models.calls) == 2


def test_generate_backoff_doubles_and_caps(monkeypatch):
    retries = 7  # enough attempts for the doubling sequence to actually hit the cap
    monkeypatch.setattr(config, "RATE_LIMIT_RETRIES", retries)
    _fake_google(monkeypatch, [_api_error(429, _MINUTE_QUOTA)])
    monkeypatch.setattr(gemini, "_pace", lambda *_a: None)  # pacing sleeps have their own tests
    sleeps: list[float] = []
    monkeypatch.setattr(gemini.time, "sleep", lambda s: sleeps.append(s))

    with pytest.raises(ExternalError):
        gemini.generate(config.GEMINI_TEXT_MODELS[0], "hi", _config(), ledger=ledger_mod.load())

    expected = []
    backoff = config.BACKOFF_START_S
    for _ in range(retries - 1):  # one sleep per retried attempt, none after the last
        expected.append(backoff)
        backoff = min(backoff * 2, config.BACKOFF_CAP_S)
    assert sleeps == expected


# ----- proactive pacing -----


@pytest.mark.parametrize("model", _one_model_per_rpm(), ids=lambda m: m.id)
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


@pytest.mark.parametrize(
    "spent_limit, first_is_still_a_candidate",
    [
        pytest.param("rpd", False, id="daily_budget_spent"),
        pytest.param("rpm", True, id="a_minute_of_requests_spent"),
    ],
)
def test_call_skips_a_model_only_once_its_daily_budget_is_spent(
    monkeypatch, spent_limit, first_is_still_a_candidate
):
    """The only test pinning which limit call() hands the ledger. Passing model.rpm instead would
    retire the 20-rpd and the 500-rpd rows alike after 5 requests and drop the run to the
    OpenRouter tier in silence, which the second row is what catches."""
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    first, second = config.GEMINI_TEXT_MODELS[0], config.GEMINI_TEXT_MODELS[1]
    ledger = ledger_mod.load()
    for _ in range(getattr(first, spent_limit)):
        ledger_mod.consume(ledger, first.id)

    assert gemini.call("s", "u", None, ledger=ledger) == "ok"
    assert models.models_tried == [first.id if first_is_still_a_candidate else second.id]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(_api_error(429, _DAY_QUOTA), id="quota_exhausted"),
        pytest.param(_api_error(500), id="external_error"),
        pytest.param(_FakeGeminiResponse("   "), id="blank_text"),
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


def test_call_returns_empty_and_says_why_when_every_model_is_spent(monkeypatch, caplog):
    """Otherwise the whole tier is skipped in silence: no dispatch line, no failure line, and a
    task that never reached the API reads exactly like one that was never asked to."""
    models = _fake_google(monkeypatch, [_FakeGeminiResponse("ok")])
    ledger = ledger_mod.load()
    for model in config.GEMINI_TEXT_MODELS:
        ledger_mod.mark_exhausted(ledger, model.id)

    with caplog.at_level(logging.WARNING, logger=gemini.logger.name):
        assert gemini.call("s", "u", None, ledger=ledger) == ""

    assert models.calls == []
    assert any("no candidate" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    "search, failure, expected",
    [
        pytest.param(False, _api_error(500), "unavailable", id="unavailable_plain"),
        pytest.param(True, _FakeGeminiResponse(""), "returned empty", id="returned_empty_grounded"),
    ],
)
def test_call_names_the_mode_in_its_warnings(monkeypatch, caplog, search, failure, expected):
    """Both modes warn from this module, so the flag is what tells a grounded line from a plain
    one. One row per warning statement, each with a different flag value, pins both call sites
    and both interpolations without re-running the cross-product."""
    model = (_search_models() if search else config.GEMINI_TEXT_MODELS)[0]
    _fake_google(monkeypatch, [failure])

    with caplog.at_level(logging.WARNING, logger=gemini.logger.name):
        gemini.call("s", "u", ledger=ledger_mod.load(), search=search)

    assert [
        r.getMessage()
        for r in caplog.records
        if f"model={model.id} search={search}" in r.getMessage() and expected in r.getMessage()
    ]


# ----- grounded search -----


@pytest.mark.parametrize("model", config.GEMINI_TEXT_MODELS, ids=lambda m: m.id)
def test_only_free_tier_grounding_models_carry_search(model):
    """Search grounding is priced out of the free tier on every 3.x model. The tool is refused
    before the request reaches a per-model quota, so the failure arrives as a 429 with no
    violation attached and leaves no usage in the console — nothing that reads as a
    misconfiguration. Flipping search onto a 3.x row is invisible until a day has passed with
    no episode."""
    assert not model.search or model.id.startswith(_FREE_GROUNDING_FAMILY)


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
