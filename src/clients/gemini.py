# src/clients/gemini.py
"""Google AI Studio transport: one paced, ledger-metered completion, and the tier loop over
the free-tier model table.

Google's free tier has no quota API, so a per-day ledger (src/core/ledger.py) plus 429s are
the only truth about what is left.
"""

import logging
import os
import time
from collections.abc import Callable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src import config
from src.core import ledger as ledger_mod
from src.core.errors import AuthError, ExternalError, QuotaExhausted
from src.core.models import ModelLimit

logger = logging.getLogger(__name__)

SOURCE = "google-ai-studio"
TEXT_LABEL = "llm google"

_QUOTA_SCOPE_MINUTE = "PerMinute"
_QUOTA_SCOPE_DAY = "PerDay"


# == Primitive ================================================================

# model id -> earliest time.monotonic() at which the next request may be dispatched
_NEXT_DISPATCH: dict[str, float] = {}
_CLIENT: genai.Client | None = None


def generate(
    model: ModelLimit,
    contents: str,
    gen_config: types.GenerateContentConfig,
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
    backoff = config.BACKOFF_START_S
    for attempt in range(max(config.RATE_LIMIT_RETRIES, 1)):
        _pace(model, contents, gen_config)
        ledger_mod.consume(ledger, model.id)
        logger.info("🚀 llm google model=%s", model.id)
        try:
            return client.models.generate_content(
                model=model.id, contents=contents, config=gen_config
            )
        except genai_errors.APIError as e:
            if e.code == config.NOT_FOUND_STATUS:
                ledger_mod.mark_exhausted(ledger, model.id)
                raise ExternalError(SOURCE, cause=e) from e
            if e.code != config.RATE_LIMIT_STATUS:
                if e.code in config.GEMINI_AUTH_STATUSES:
                    raise AuthError(SOURCE, cause=e) from e
                raise ExternalError(SOURCE, cause=e) from e
            if _quota_scope(e) != _QUOTA_SCOPE_DAY and attempt < config.RATE_LIMIT_RETRIES - 1:
                logger.warning("⚠️ llm google model=%s rate-limited; backoff %ss", model.id, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, config.BACKOFF_CAP_S)
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
        key = os.environ.get(config.GEMINI_KEY_LABEL)
        if not key:
            raise AuthError(SOURCE, detail=f"{config.GEMINI_KEY_LABEL} unset")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


def _pace(model: ModelLimit, contents: str, gen_config: types.GenerateContentConfig) -> None:
    """Sleep until the model's rpm/tpm window allows another request, then reserve the next
    slot — 429 is the exception path, not the pacing mechanism."""
    wait = _NEXT_DISPATCH.get(model.id, 0.0) - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    est_tokens = len(contents) // 4 + (
        gen_config.max_output_tokens or config.GEMINI_EST_OUTPUT_TOKENS
    )
    _NEXT_DISPATCH[model.id] = time.monotonic() + max(60 / model.rpm, est_tokens / model.tpm * 60)


def _quota_scope(e: genai_errors.APIError) -> str | None:
    """Which quota window a 429 names, read from its violation details (None when unstated)."""
    details = str(getattr(e, "details", ""))
    for scope in (_QUOTA_SCOPE_DAY, _QUOTA_SCOPE_MINUTE):
        if scope in details:
            return scope
    return None


# == Tier =====================================================================


def first_completion(
    models: list[ModelLimit],
    contents: str,
    gen_config: types.GenerateContentConfig,
    *,
    ledger: ledger_mod.Ledger,
    label: str,
    on_response: Callable[[types.GenerateContentResponse], None] | None = None,
) -> str:
    """First non-empty text from `models` in order; "" when every candidate is spent, failing
    or empty.

    The sole owner of the tier policy — which candidates the ledger still allows, what a failure
    costs the run, and when the cascade is dry — so a change to it reaches every caller. Callers
    differ only in their model subset, their `label` (which names the cascade in the warnings)
    and `on_response`, a hook for reading the raw response before its text is taken.

    QuotaExhausted arrives as an ExternalError subclass and is tolerated with it. AuthError
    propagates — the cross-tier decision belongs to the cascade in src/clients/llm.py."""
    for model in models:
        if not ledger_mod.available(ledger, model.id, model.rpd):
            continue
        try:
            response = generate(model, contents, gen_config, ledger=ledger)
        except AuthError:
            raise
        except ExternalError as e:
            logger.warning("⚠️ %s model=%s unavailable, next candidate: %s", label, model.id, e)
            continue
        if on_response is not None:
            on_response(response)
        text = response.text
        if text and text.strip():
            return text
        logger.warning("⚠️ %s model=%s returned empty, next candidate", label, model.id)
    return ""


def call(system: str, user: str, max_tokens: int | None, *, ledger: ledger_mod.Ledger) -> str:
    """First non-empty completion from the general text model table.

    Plain generation with no grounding tool — search is podcast research's alone."""
    gen_config = types.GenerateContentConfig(
        system_instruction=system, max_output_tokens=max_tokens
    )
    return first_completion(
        config.GEMINI_TEXT_MODELS, user, gen_config, ledger=ledger, label=TEXT_LABEL
    )
