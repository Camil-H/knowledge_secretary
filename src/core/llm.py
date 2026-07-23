"""OpenRouter-only LLM calls: live free-model selection + rate-limit backoff."""

import logging
import time

import httpx
import litellm

from src.core.errors import AuthError, ExternalError

logger = logging.getLogger(__name__)

litellm.drop_params = True  # tolerate provider-specific unsupported params

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
RANK_OUTPUT = "max_output_tokens"
RANK_CONTEXT = "context"
_FREE_LIMIT = 8
_HTTP_TIMEOUT_S = 20
_RATE_LIMIT_RETRIES = 4
_BACKOFF_START_S = 2
_BACKOFF_CAP_S = 30
FALLBACK_MODEL = "openrouter/deepseek/deepseek-chat-v3-0324:free"


# == Model resolution =========================================================


def _free_openrouter_models(rank: str, *, limit: int = _FREE_LIMIT) -> list[str]:
    """Live-fetch zero-cost OpenRouter models, ranked for the use case ([] on failure)."""
    try:
        data = httpx.get(OPENROUTER_MODELS_URL, timeout=_HTTP_TIMEOUT_S).json()["data"]
    except (httpx.HTTPError, ValueError, KeyError) as e:
        logger.warning("⚠️ openrouter model list degraded: %s", e)
        return []

    free = [
        m
        for m in data
        if str(m.get("pricing", {}).get("prompt")) == "0"
        and str(m.get("pricing", {}).get("completion")) == "0"
    ]

    def _rank_key(m: dict) -> int:
        if rank == RANK_OUTPUT:
            return (m.get("top_provider") or {}).get("max_completion_tokens") or 0
        return m.get("context_length") or 0

    free.sort(key=_rank_key, reverse=True)
    return [f"openrouter/{m['id']}" for m in free[:limit]]


def resolve_models(podcast: bool | None = None) -> list[str]:
    """Ranked zero-cost models — by output tokens for the podcast, else by context."""
    return _free_openrouter_models(RANK_OUTPUT if podcast else RANK_CONTEXT)


# == Completion ===============================================================


def call(system: str, user: str, *, max_tokens: int | None = None) -> str:
    """First non-empty completion across the resolved models: 429 retries the same
    model with backoff, an auth failure raises immediately, other errors fall
    through, all-fail raises ExternalError."""
    models = resolve_models() or [FALLBACK_MODEL]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Exception | None = None
    for model in models:
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
                    logger.warning("⚠️ llm model=%s rate-limited; backoff %ss", model, backoff)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_CAP_S)
                    continue
                logger.warning("⚠️ llm model=%s unavailable, next candidate: %s", model, e)
                break

    raise ExternalError("openrouter", detail="all models failed", cause=last_err)


# == Helper Functions =========================================================


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
    return "no user or org" in msg or "auth" in msg
