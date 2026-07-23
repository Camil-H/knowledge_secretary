"""OpenRouter-only LLM calls: live free-model selection + rate-limit backoff."""

import logging
import os
import time
from typing import Any

import httpx
import litellm

from src.core.errors import AuthError, ExternalError

logger = logging.getLogger(__name__)

# One entry from the OpenRouter /models catalog; only a few keys are read.
type ModelRecord = dict[str, Any]

litellm.drop_params = True  # tolerate provider-specific unsupported params

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
RANK_OUTPUT = "max_output_tokens"
RANK_CONTEXT = "context"
_FREE_LIMIT = int(os.environ.get("LLM_FREE_LIMIT", "8"))
_HTTP_TIMEOUT_S = 20
_RATE_LIMIT_RETRIES = int(os.environ.get("LLM_RATE_LIMIT_RETRIES", "4"))
_BACKOFF_START_S = 2
_BACKOFF_CAP_S = 30
# wall-clock budget for the whole model cascade, so a many-item run can't burn
# tens of minutes purely on backoff sleep (monotonic, immune to clock changes).
_DEADLINE_S = float(os.environ.get("LLM_DEADLINE_S", "120"))
FALLBACK_MODEL = "openrouter/google/gemma-4-31b-it:free"
# unambiguous auth-failure phrasing only — a bare "auth" substring also matches "author" etc.
_AUTH_PHRASES = ("no user or org", "invalid api key", "unauthorized")
# ids passing the zero-price filter that aren't general text writers (music / guardrail / router)
_EXCLUDE_IDS = ("lyria", "content-safety", "openrouter/free")

# Curated known-good free models, best first. Layered on top of the live ranking in
# resolve_models(): a preferred id absent from the current live list is simply skipped.
PREFERRED_CONTEXT = [
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/openai/gpt-oss-20b:free",
]
PREFERRED_OUTPUT = [
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/cohere/north-mini-code:free",
    "openrouter/openai/gpt-oss-20b:free",
]


# == Model resolution =========================================================

# rank mode -> resolved live ranking, memoized for the process lifetime so the
# catalog is fetched at most once per mode per run (see _reset_model_cache for tests).
_MODEL_CACHE: dict[str, list[str]] = {}


def _reset_model_cache() -> None:
    """Clear the memoized live rankings (test-only escape hatch)."""
    _MODEL_CACHE.clear()


def _free_openrouter_models(rank: str, *, limit: int = _FREE_LIMIT) -> list[str]:
    """Live-fetch zero-cost OpenRouter models, ranked for the use case ([] on failure).

    Memoized per rank mode; a failed fetch is not cached, so a later call can retry."""
    if rank in _MODEL_CACHE:
        return _MODEL_CACHE[rank]

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

    def _rank_key(m: ModelRecord) -> int:
        if rank == RANK_OUTPUT:
            return (m.get("top_provider") or {}).get("max_completion_tokens") or 0
        return m.get("context_length") or 0

    free.sort(key=_rank_key, reverse=True)
    result = [f"openrouter/{m['id']}" for m in free[:limit]]
    _MODEL_CACHE[rank] = result
    return result


def resolve_models(podcast: bool | None = None) -> list[str]:
    """Ranked zero-cost models — by output tokens for the podcast, else by context.

    Curated PREFERRED ids present in the live-ranked list lead, in preferred order;
    the rest of the live ranking follows, deduped. A preferred id currently unavailable
    for free is silently skipped, and an empty live fetch still yields []."""
    live = _free_openrouter_models(RANK_OUTPUT if podcast else RANK_CONTEXT)
    preferred = PREFERRED_OUTPUT if podcast else PREFERRED_CONTEXT
    live_set = set(live)
    ordered_preferred = [m for m in preferred if m in live_set]
    preferred_set = set(ordered_preferred)
    remaining = [m for m in live if m not in preferred_set]
    return ordered_preferred + remaining


# == Completion ===============================================================


def call(system: str, user: str, *, max_tokens: int | None = None) -> str:
    """First non-empty completion across the resolved models: 429 retries the same
    model with backoff, an auth failure raises immediately, other errors fall
    through, all-fail raises ExternalError. A wall-clock deadline caps total
    time so the cascade is abandoned rather than walking every model x retry."""
    models = resolve_models() or [FALLBACK_MODEL]
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
                resp = litellm.completion(model=model, messages=messages, max_tokens=max_tokens)
                content = resp.choices[0].message.content
                if content and content.strip():
                    return content
                logger.warning("⚠️ llm model=%s returned empty, next candidate", model)
                last_err = RuntimeError(f"{model} returned empty content")
                break
            except Exception as e:
                last_err = e
                if _is_auth(e):
                    raise AuthError("openrouter", cause=e) from e
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

    raise ExternalError("openrouter", detail="all models failed", cause=last_err)


# == Helper Functions =========================================================


def _writes_text(model: ModelRecord) -> bool:
    """Exclude free ids that pass the price filter but aren't general text writers
    (music/guardrail/router models, or non-text output)."""
    if any(bad in model.get("id", "") for bad in _EXCLUDE_IDS):
        return False
    out = (model.get("architecture") or {}).get("output_modalities")
    return not out or "text" in out


def _is_rate_limit(e: Exception) -> bool:
    """True for 429 / rate-limit errors (across litellm and raw provider errors)."""
    if isinstance(e, getattr(litellm, "RateLimitError", ())):
        return True
    if getattr(e, "status_code", None) == 429:
        return True
    return "rate limit" in str(e).lower()


def _is_auth(e: Exception) -> bool:
    """True for credential / 401 errors (across litellm and raw provider errors)."""
    if isinstance(e, getattr(litellm, "AuthenticationError", ())):
        return True
    if getattr(e, "status_code", None) == 401:
        return True
    msg = str(e).lower()
    return any(phrase in msg for phrase in _AUTH_PHRASES)
