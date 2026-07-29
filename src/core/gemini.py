# src/core/gemini.py
"""Google AI Studio transport: one paced, ledger-metered completion, and the tier loop over
the free-tier model table.

Google's free tier has no quota API, so a per-day ledger (src/core/ledger.py) plus 429s are
the only truth about what is left.
"""

import logging
import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.core import ledger as ledger_mod
from src.core.errors import AuthError, ExternalError, QuotaExhausted

logger = logging.getLogger(__name__)

KEY_LABEL = "GOOGLE_AI_STUDIO_KEY"
SOURCE = "google-ai-studio"

_RATE_LIMIT_RETRIES = int(os.environ.get("LLM_RATE_LIMIT_RETRIES", "4"))
_BACKOFF_START_S = 2
_BACKOFF_CAP_S = 30
_UNAUTHORIZED_STATUS = 401
_AUTH_STATUSES = (_UNAUTHORIZED_STATUS, 403)


@dataclass(frozen=True)
class ModelLimit:
    """An AI Studio model with its free-tier ceilings: requests/day, requests/min, tokens/min."""

    id: str
    rpd: int
    rpm: int
    tpm: int


# RPD is per model, so this table is a ~60 request/day pool. rpm/tpm drive proactive pacing;
# the SDK exposes no quota fields, so nothing here is ever polled.
TEXT_MODELS: list[ModelLimit] = [
    ModelLimit("gemini-3.6-flash", rpd=20, rpm=5, tpm=250_000),
    ModelLimit("gemini-3.6-flash-lite", rpd=20, rpm=5, tpm=250_000),
    ModelLimit("gemini-3.1-flash", rpd=20, rpm=5, tpm=250_000),
]
_EST_OUTPUT_TOKENS = 2048
_QUOTA_SCOPE_MINUTE = "PerMinute"
_QUOTA_SCOPE_DAY = "PerDay"
_RATE_LIMIT_STATUS = 429
_NOT_FOUND_STATUS = 404


# == Primitive ================================================================

# model id -> earliest time.monotonic() at which the next request may be dispatched
_NEXT_DISPATCH: dict[str, float] = {}
_CLIENT: genai.Client | None = None


def generate(
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
    client = _client()
    backoff = _BACKOFF_START_S
    for attempt in range(max(_RATE_LIMIT_RETRIES, 1)):
        _pace(model, contents, config)
        ledger_mod.consume(ledger, model.id)
        logger.info("🚀 llm google model=%s", model.id)
        try:
            return client.models.generate_content(model=model.id, contents=contents, config=config)
        except genai_errors.APIError as e:
            if e.code == _NOT_FOUND_STATUS:
                # a model id that does not exist would otherwise be re-paced and re-dispatched
                # on every call for the rest of the day
                ledger_mod.mark_exhausted(ledger, model.id)
                raise ExternalError(SOURCE, cause=e) from e
            if e.code != _RATE_LIMIT_STATUS:
                if e.code in _AUTH_STATUSES:
                    raise AuthError(SOURCE, cause=e) from e
                raise ExternalError(SOURCE, cause=e) from e
            if _quota_scope(e) != _QUOTA_SCOPE_DAY and attempt < _RATE_LIMIT_RETRIES - 1:
                logger.warning("⚠️ llm google model=%s rate-limited; backoff %ss", model.id, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_CAP_S)
                continue
            ledger_mod.mark_exhausted(ledger, model.id)
            raise QuotaExhausted(model.id, cause=e) from e
        except Exception as e:
            raise ExternalError(SOURCE, cause=e) from e

    raise ExternalError(SOURCE, detail=f"{model.id} retries exhausted")


def _client() -> genai.Client:
    """The memoized AI Studio client; a missing key is an auth failure, not a KeyError."""
    global _CLIENT
    if _CLIENT is None:
        key = os.environ.get(KEY_LABEL)
        if not key:
            raise AuthError(SOURCE, detail=f"{KEY_LABEL} unset")
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


def _quota_scope(e: genai_errors.APIError) -> str | None:
    """Which quota window a 429 names, read from its violation details (None when unstated)."""
    details = str(getattr(e, "details", ""))
    for scope in (_QUOTA_SCOPE_DAY, _QUOTA_SCOPE_MINUTE):
        if scope in details:
            return scope
    return None


# == Tier =====================================================================


def call(system: str, user: str, max_tokens: int | None, *, ledger: ledger_mod.Ledger) -> str:
    """First non-empty completion from the model table; "" when every model is spent, failing
    or empty.

    AuthError propagates — the cross-tier decision belongs to the cascade in src/core/llm.py."""
    config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens)
    for model in TEXT_MODELS:
        if not ledger_mod.available(ledger, model.id, model.rpd):
            continue
        try:
            response = generate(model, user, config, ledger=ledger)
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
