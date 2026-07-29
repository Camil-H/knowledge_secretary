# src/clients/openrouter.py
"""OpenRouter transport: the live zero-cost model catalog, one chat completion, and the
deadline-bounded tier loop over the ranked candidates.

Its key is independent of the Google one on purpose, so this tier still answers when the
AI Studio credential is broken.
"""

import logging
import os
import time
from typing import Any

import httpx

from src import config
from src.core.errors import AuthError, ExternalError

logger = logging.getLogger(__name__)

# One entry from the OpenRouter /models catalog; only a few keys are read.
type ModelRecord = dict[str, Any]

SOURCE = "openrouter"
_CASCADE_SOURCE = "llm"

MODELS_URL = "https://openrouter.ai/api/v1/models"
COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_PREFIX = "openrouter/"
_AUTH_PHRASES = ("no user or org", "invalid api key", "unauthorized")
_MAX_ERROR_BODY_CHARS = 300


# == Exceptions ===============================================================


class OpenRouterError(Exception):
    """A non-OK OpenRouter response; carries the status and a truncated error body."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        super().__init__(f"openrouter {status_code}: {body}")


# == Tier =====================================================================


def call(system: str, user: str, max_tokens: int | None) -> str:
    """First non-empty OpenRouter completion: 429 retries the same model with backoff, an
    auth failure raises immediately, other errors fall through, all-fail raises ExternalError.
    A wall-clock deadline caps total time so the cascade is abandoned rather than walking
    every model x retry."""
    candidates = models()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    deadline = time.monotonic() + config.OPENROUTER_DEADLINE_S
    last_err: Exception | None = None
    for model in candidates:
        if time.monotonic() >= deadline:
            logger.warning(
                "⚠️ llm deadline %ss reached, abandoning cascade", config.OPENROUTER_DEADLINE_S
            )
            break
        logger.info("🚀 llm model=%s", model)
        backoff = config.BACKOFF_START_S
        for attempt in range(config.RATE_LIMIT_RETRIES):
            try:
                content = complete(model, messages, max_tokens)
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
                    raise AuthError(SOURCE, cause=e) from e
                if _is_rate_limit(e) and attempt < config.RATE_LIMIT_RETRIES - 1:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break  # outer deadline guard abandons the cascade next iteration
                    logger.warning("⚠️ llm model=%s rate-limited; backoff %ss", model, backoff)
                    time.sleep(min(backoff, remaining))
                    backoff = min(backoff * 2, config.BACKOFF_CAP_S)
                    continue
                logger.warning(
                    "⚠️ llm model=%s unavailable, next candidate: %s status=%s",
                    model,
                    type(e).__name__,
                    getattr(e, "status_code", None),
                )
                break

    raise ExternalError(_CASCADE_SOURCE, detail="all models failed", cause=last_err)


def complete(model: str, messages: list[dict[str, str]], max_tokens: int | None) -> str:
    """The assistant text of one OpenRouter chat completion ("" when it returns none).

    A non-OK response raises OpenRouterError, which the tier loop classifies."""
    key = os.environ.get(config.OPENROUTER_KEY_LABEL)
    if not key:
        raise AuthError(SOURCE, detail=f"{config.OPENROUTER_KEY_LABEL} unset")
    response = httpx.post(
        COMPLETIONS_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model.removeprefix(_PREFIX),
            "messages": messages,
            "max_tokens": max_tokens,
        },
        timeout=config.HTTP_TIMEOUT_S,
    )
    if response.status_code >= config.ERROR_STATUS_FLOOR:
        raise OpenRouterError(response.status_code, response.text[:_MAX_ERROR_BODY_CHARS])
    choices = response.json().get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    return message.get("content") or ""


# == Model resolution =========================================================

_MODEL_CACHE: list[str] = []


def _reset_model_cache() -> None:
    """Clear the memoized live ranking (test-only escape hatch)."""
    _MODEL_CACHE.clear()


def models() -> list[str]:
    """Zero-cost models, context-ranked, with the curated preferred ids leading.

    A preferred id currently unavailable for free is silently skipped. When the catalog cannot
    be fetched the curated list stands in unranked, so an unreachable catalog costs ranking
    rather than the whole tier."""
    live = _free_models()
    if not live:
        return list(config.OPENROUTER_PREFERRED_CONTEXT)
    live_set = set(live)
    ordered_preferred = [m for m in config.OPENROUTER_PREFERRED_CONTEXT if m in live_set]
    preferred_set = set(ordered_preferred)
    return ordered_preferred + [m for m in live if m not in preferred_set]


def _free_models(*, limit: int = config.OPENROUTER_FREE_LIMIT) -> list[str]:
    """Live-fetch zero-cost OpenRouter models, ranked by context length ([] on failure).

    Memoized per process; a failed fetch is not cached, so a later call can retry."""
    if _MODEL_CACHE:
        return _MODEL_CACHE

    try:
        catalog = httpx.get(MODELS_URL, timeout=config.HTTP_TIMEOUT_S).json()
        data: list[ModelRecord] = catalog["data"]
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
    _MODEL_CACHE[:] = [f"{_PREFIX}{m['id']}" for m in free[:limit]]
    return _MODEL_CACHE


# == Helper Functions =========================================================


def _writes_text(model: ModelRecord) -> bool:
    """Exclude free ids that pass the price filter but aren't general text writers
    (music/guardrail/router models, or non-text output)."""
    if any(bad in model.get("id", "") for bad in config.OPENROUTER_EXCLUDE_IDS):
        return False
    out = (model.get("architecture") or {}).get("output_modalities")
    return not out or "text" in out


def _is_rate_limit(e: Exception) -> bool:
    """True for 429 / rate-limit responses from OpenRouter."""
    if getattr(e, "status_code", None) == config.RATE_LIMIT_STATUS:
        return True
    return "rate limit" in str(e).lower()


def _is_auth(e: Exception) -> bool:
    """True for credential / 401 failures from OpenRouter."""
    if getattr(e, "status_code", None) == config.UNAUTHORIZED_STATUS:
        return True
    msg = str(e).lower()
    return any(phrase in msg for phrase in _AUTH_PHRASES)
