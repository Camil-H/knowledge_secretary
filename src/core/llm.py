# src/core/llm.py
"""LLM transport: the Google AI Studio free tier first, OpenRouter's free models behind it.

The two tiers are separate credentials on purpose — a broken Google key degrades to
OpenRouter instead of taking the newsletter, the youtube digest and the podcast down at once.
Google's free tier has no quota API, so a per-day ledger (src/core/ledger.py) plus 429s are
the only truth about what is left.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.core import ledger as ledger_mod
from src.core.errors import AuthError, ExternalError, QuotaExhausted

logger = logging.getLogger(__name__)

# One entry from the OpenRouter /models catalog; only a few keys are read.
type ModelRecord = dict[str, Any]

GOOGLE_KEY_LABEL = "GOOGLE_AI_STUDIO_KEY"
OPENROUTER_KEY_LABEL = "OPENROUTER_API_KEY"
GOOGLE_SOURCE = "google-ai-studio"
OPENROUTER_SOURCE = "openrouter"
LLM_SOURCE = "llm"

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_PREFIX = "openrouter/"
_FREE_LIMIT = int(os.environ.get("LLM_FREE_LIMIT", "8"))
_HTTP_TIMEOUT_S = 20
_RATE_LIMIT_RETRIES = int(os.environ.get("LLM_RATE_LIMIT_RETRIES", "4"))
_BACKOFF_START_S = 2
_BACKOFF_CAP_S = 30
# wall-clock cap for the whole cascade so a many-item run can't burn minutes on backoff sleep
_DEADLINE_S = float(os.environ.get("LLM_DEADLINE_S", "120"))
FALLBACK_MODEL = "openrouter/google/gemma-4-31b-it:free"
_AUTH_PHRASES = ("no user or org", "invalid api key", "unauthorized")
_UNAUTHORIZED_STATUS = 401
_AUTH_STATUSES = (_UNAUTHORIZED_STATUS, 403)
_ERROR_STATUS_FLOOR = 400
# ids passing the zero-price filter that aren't general text writers (music / guardrail / router)
_EXCLUDE_IDS = ("lyria", "content-safety", "openrouter/free")
_MAX_ERROR_BODY_CHARS = 300

# Curated known-good free models, best first. Layered on top of the live ranking in
# _openrouter_models(): a preferred id absent from the current live list is simply skipped.
PREFERRED_CONTEXT = [
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/openai/gpt-oss-20b:free",
]


@dataclass(frozen=True)
class ModelLimit:
    """An AI Studio model with its free-tier ceilings: requests/day, requests/min, tokens/min."""

    id: str
    rpd: int
    rpm: int
    tpm: int


# RPD is per model, so this table is a ~60 request/day pool. rpm/tpm drive proactive pacing;
# the SDK exposes no quota fields, so nothing here is ever polled.
GEMINI_TEXT_MODELS: list[ModelLimit] = [
    ModelLimit("gemini-3.6-flash", rpd=20, rpm=5, tpm=250_000),
    ModelLimit("gemini-3.6-flash-lite", rpd=20, rpm=5, tpm=250_000),
    ModelLimit("gemini-3.1-flash", rpd=20, rpm=5, tpm=250_000),
]
_EST_OUTPUT_TOKENS = 2048
_QUOTA_SCOPE_MINUTE = "PerMinute"
_QUOTA_SCOPE_DAY = "PerDay"
_RATE_LIMIT_STATUS = 429


# == Exceptions ===============================================================


class OpenRouterError(Exception):
    """A non-OK OpenRouter response; carries the status and a truncated error body."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        super().__init__(f"openrouter {status_code}: {body}")


# == Google AI Studio primitive ===============================================

# model id -> earliest time.monotonic() at which the next request may be dispatched
_NEXT_DISPATCH: dict[str, float] = {}
_CLIENT: genai.Client | None = None


def gemini_generate(
    model: ModelLimit,
    contents: str,
    config: types.GenerateContentConfig,
    *,
    ledger: ledger_mod.Ledger,
) -> types.GenerateContentResponse:
    """One AI Studio completion, paced to the model's own rpm/tpm and counted in the ledger.

    The budget is spent at dispatch and never refunded — the provider may have counted an
    attempt that failed for us. A per-minute 429 is retried with capped exponential backoff;
    a per-day 429 (or one that outlives the retries) retires the model for the day and raises
    QuotaExhausted. A 401/403 raises AuthError, anything else ExternalError; the raise is the
    terminal signal, so no failure is logged here.

    Returns the raw response, not its text, so callers can read grounding metadata."""
    backoff = _BACKOFF_START_S
    for attempt in range(max(_RATE_LIMIT_RETRIES, 1)):
        client = _google_client()
        _pace(model, contents, config)
        ledger_mod.consume(ledger, model.id)
        logger.info("🚀 llm google model=%s", model.id)
        try:
            return client.models.generate_content(model=model.id, contents=contents, config=config)
        except genai_errors.APIError as e:
            if e.code != _RATE_LIMIT_STATUS:
                raise _typed_google_error(e) from e
            if _quota_scope(e) != _QUOTA_SCOPE_DAY and attempt < _RATE_LIMIT_RETRIES - 1:
                logger.warning("⚠️ llm google model=%s rate-limited; backoff %ss", model.id, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_CAP_S)
                continue
            ledger_mod.mark_exhausted(ledger, model.id)
            raise QuotaExhausted(model.id, cause=e) from e
        except Exception as e:
            raise ExternalError(GOOGLE_SOURCE, cause=e) from e

    raise ExternalError(GOOGLE_SOURCE, detail=f"{model.id} retries exhausted")


def _google_client() -> genai.Client:
    """The memoized AI Studio client; a missing key is an auth failure, not a KeyError."""
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get(GOOGLE_KEY_LABEL)
        if not key:
            raise AuthError(GOOGLE_SOURCE, detail=f"{GOOGLE_KEY_LABEL} unset")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


def _pace(model: ModelLimit, contents: str, config: types.GenerateContentConfig) -> None:
    """Sleep until the model's rpm/tpm window allows another request, then reserve the next
    slot — 429 is the exception path, not the pacing mechanism."""
    wait = _NEXT_DISPATCH.get(model.id, 0.0) - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    est_tokens = len(contents) // 4 + (config.max_output_tokens or _EST_OUTPUT_TOKENS)
    _NEXT_DISPATCH[model.id] = time.monotonic() + max(60 / model.rpm, est_tokens / model.tpm * 60)


def _typed_google_error(e: genai_errors.APIError) -> ExternalError:
    if e.code in _AUTH_STATUSES:
        return AuthError(GOOGLE_SOURCE, cause=e)
    return ExternalError(GOOGLE_SOURCE, cause=e)


def _quota_scope(e: genai_errors.APIError) -> str | None:
    """Which quota window a 429 names, read from its violation details (None when unstated)."""
    details = str(getattr(e, "details", ""))
    for scope in (_QUOTA_SCOPE_DAY, _QUOTA_SCOPE_MINUTE):
        if scope in details:
            return scope
    return None


# == Cascade ==================================================================


def call(system: str, user: str, *, max_tokens: int | None = None) -> str:
    """First non-empty completion from the Google tier, then the OpenRouter tier.

    All tiers dry raises ExternalError. An OpenRouter auth failure propagates as AuthError;
    a Google one does not, because OpenRouter is an independent credential."""
    ledger = ledger_mod.load()
    try:
        text = _gemini_tier(system, user, max_tokens, ledger)
    except AuthError as e:
        # deliberate cross-tier degradation: one bad Google key must not down every product
        logger.warning("⚠️ llm google tier unavailable, degrading to openrouter: %s", e)
        text = ""
    return text or _openrouter_tier(system, user, max_tokens)


def _gemini_tier(system: str, user: str, max_tokens: int | None, ledger: ledger_mod.Ledger) -> str:
    """First non-empty Gemini completion; "" when every model is spent, failing or empty.

    AuthError propagates — the cross-tier decision belongs to call()."""
    config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens)
    for model in GEMINI_TEXT_MODELS:
        if not ledger_mod.available(ledger, model.id, model.rpd):
            continue
        try:
            response = gemini_generate(model, user, config, ledger=ledger)
        except AuthError:
            raise
        except ExternalError as e:
            logger.warning("⚠️ llm google model=%s unavailable, next candidate: %s", model.id, e)
            continue
        text = response.text
        if text and text.strip():
            return text
        logger.warning("⚠️ llm google model=%s returned empty, next candidate", model.id)
    return ""


def _openrouter_tier(system: str, user: str, max_tokens: int | None) -> str:
    """First non-empty OpenRouter completion: 429 retries the same model with backoff, an
    auth failure raises immediately, other errors fall through, all-fail raises ExternalError.
    A wall-clock deadline caps total time so the cascade is abandoned rather than walking
    every model x retry."""
    models = _openrouter_models() or [FALLBACK_MODEL]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    deadline = time.monotonic() + _DEADLINE_S
    last_err: Exception | None = None
    for model in models:
        if time.monotonic() >= deadline:
            logger.warning("⚠️ llm deadline %ss reached, abandoning cascade", _DEADLINE_S)
            break
        logger.info("🚀 llm model=%s", model)
        backoff = _BACKOFF_START_S
        for attempt in range(_RATE_LIMIT_RETRIES):
            try:
                content = _openrouter_complete(model, messages, max_tokens)
                if content and content.strip():
                    return content
                logger.warning("⚠️ llm model=%s returned empty, next candidate", model)
                last_err = RuntimeError(f"{model} returned empty content")
                break
            except AuthError:
                raise
            except Exception as e:
                last_err = e
                if _is_auth(e):
                    raise AuthError(OPENROUTER_SOURCE, cause=e) from e
                if _is_rate_limit(e) and attempt < _RATE_LIMIT_RETRIES - 1:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break  # outer deadline guard abandons the cascade next iteration
                    logger.warning("⚠️ llm model=%s rate-limited; backoff %ss", model, backoff)
                    time.sleep(min(backoff, remaining))
                    backoff = min(backoff * 2, _BACKOFF_CAP_S)
                    continue
                logger.warning(
                    "⚠️ llm model=%s unavailable, next candidate: %s status=%s",
                    model,
                    type(e).__name__,
                    getattr(e, "status_code", None),
                )
                break

    raise ExternalError(LLM_SOURCE, detail="all models failed", cause=last_err)


def _openrouter_complete(model: str, messages: list[dict[str, str]], max_tokens: int | None) -> str:
    """The assistant text of one OpenRouter chat completion ("" when it returns none).

    A non-OK response raises OpenRouterError, which the cascade classifies."""
    key = os.environ.get(OPENROUTER_KEY_LABEL)
    if not key:
        raise AuthError(OPENROUTER_SOURCE, detail=f"{OPENROUTER_KEY_LABEL} unset")
    response = httpx.post(
        OPENROUTER_COMPLETIONS_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model.removeprefix(_OPENROUTER_PREFIX),
            "messages": messages,
            "max_tokens": max_tokens,
        },
        timeout=_HTTP_TIMEOUT_S,
    )
    if response.status_code >= _ERROR_STATUS_FLOOR:
        raise OpenRouterError(response.status_code, response.text[:_MAX_ERROR_BODY_CHARS])
    choices = response.json().get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    return message.get("content") or ""


# == Model resolution =========================================================

# live context ranking, memoized per process (see _reset_model_cache for tests)
_MODEL_CACHE: list[str] = []


def _reset_model_cache() -> None:
    """Clear the memoized live ranking (test-only escape hatch)."""
    _MODEL_CACHE.clear()


def _openrouter_models() -> list[str]:
    """Zero-cost models, context-ranked, with the curated PREFERRED_CONTEXT ids leading.

    A preferred id currently unavailable for free is silently skipped, and an empty live
    fetch still yields []."""
    live = _free_openrouter_models()
    live_set = set(live)
    ordered_preferred = [m for m in PREFERRED_CONTEXT if m in live_set]
    preferred_set = set(ordered_preferred)
    return ordered_preferred + [m for m in live if m not in preferred_set]


def _free_openrouter_models(*, limit: int = _FREE_LIMIT) -> list[str]:
    """Live-fetch zero-cost OpenRouter models, ranked by context length ([] on failure).

    Memoized per process; a failed fetch is not cached, so a later call can retry."""
    if _MODEL_CACHE:
        return _MODEL_CACHE

    try:
        data: list[ModelRecord] = httpx.get(OPENROUTER_MODELS_URL, timeout=_HTTP_TIMEOUT_S).json()[
            "data"
        ]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        logger.warning(
            "⚠️ openrouter model list degraded: %s status=%s",
            type(e).__name__,
            getattr(e, "status_code", None),
        )
        return []

    free = [
        m
        for m in data
        if str(m.get("pricing", {}).get("prompt")) == "0"
        and str(m.get("pricing", {}).get("completion")) == "0"
        and _writes_text(m)
    ]
    free.sort(key=lambda m: m.get("context_length") or 0, reverse=True)
    _MODEL_CACHE[:] = [f"{_OPENROUTER_PREFIX}{m['id']}" for m in free[:limit]]
    return _MODEL_CACHE


# == Helper Functions =========================================================


def _writes_text(model: ModelRecord) -> bool:
    """Exclude free ids that pass the price filter but aren't general text writers
    (music/guardrail/router models, or non-text output)."""
    if any(bad in model.get("id", "") for bad in _EXCLUDE_IDS):
        return False
    out = (model.get("architecture") or {}).get("output_modalities")
    return not out or "text" in out


def _is_rate_limit(e: Exception) -> bool:
    """True for 429 / rate-limit responses from OpenRouter."""
    if getattr(e, "status_code", None) == _RATE_LIMIT_STATUS:
        return True
    return "rate limit" in str(e).lower()


def _is_auth(e: Exception) -> bool:
    """True for credential / 401 failures from OpenRouter."""
    if getattr(e, "status_code", None) == _UNAUTHORIZED_STATUS:
        return True
    msg = str(e).lower()
    return any(phrase in msg for phrase in _AUTH_PHRASES)
