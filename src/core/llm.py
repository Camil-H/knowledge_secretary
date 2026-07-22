"""OpenRouter-only LLM calls: dynamic free-model selection + rate-limit backoff.

Models are the live zero-cost OpenRouter catalog (ids ending in `:free`), ranked
by context_length — or by max_completion_tokens when resolving for the podcast,
which needs long OUTPUT. LiteLLM reads the key from OPENROUTER_API_KEY. Free
models share a fixed ~20 RPM account cap, so a 429 is retried with capped
exponential backoff (switching models can't clear an account-wide limit); other
errors fall through.
"""

import logging
import time

import httpx
import litellm

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
# used when the live free list is empty
FALLBACK_MODEL = "openrouter/deepseek/deepseek-chat-v3-0324:free"


# == Model resolution =========================================================


def _free_openrouter_models(rank: str, *, limit: int = _FREE_LIMIT) -> list[str]:
    """Live-fetch zero-cost OpenRouter models, ranked for the use case.

    Degrades to [] (a missing free list is non-fatal — call() falls back to
    FALLBACK_MODEL), so this is logged as a warning rather than raised.
    """
    try:
        data = httpx.get(OPENROUTER_MODELS_URL, timeout=_HTTP_TIMEOUT_S).json()["data"]
    except Exception as e:
        logger.warning("⚠️ openrouter model list unavailable, using fallback: %s", e)
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
    """Ranked zero-cost OpenRouter models. The podcast ranks by max_completion_tokens
    (it needs long OUTPUT); everything else ranks by context length."""
    return _free_openrouter_models(RANK_OUTPUT if podcast else RANK_CONTEXT)


# == Completion ===============================================================


def call(task: str, system: str, user: str, *, max_tokens: int | None = None) -> str:
    """Try each resolved model in order; return the first non-empty completion.

    A 429 retries the same model with capped exponential backoff; any other error
    falls through to the next candidate. Raises RuntimeError only if every
    candidate fails — the caller decides whether to tolerate that and logs it.
    """
    models = resolve_models() or [FALLBACK_MODEL]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Exception | None = None
    for model in models:
        backoff = _BACKOFF_START_S
        for attempt in range(_RATE_LIMIT_RETRIES):
            try:
                resp = litellm.completion(model=model, messages=messages, max_tokens=max_tokens)
                content = resp.choices[0].message.content
                if content and content.strip():
                    return content
                last_err = RuntimeError(f"{model} returned empty content")
                break  # empty content: try the next model, don't retry this one
            except Exception as e:
                last_err = e
                if _is_rate_limit(e) and attempt < _RATE_LIMIT_RETRIES - 1:
                    logger.warning(
                        "⚠️ llm tier=%s model=%s rate-limited; backoff %ss", task, model, backoff
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, _BACKOFF_CAP_S)
                    continue
                logger.warning("⚠️ llm tier=%s model=%s unavailable, next candidate", task, model)
                break

    raise RuntimeError(f"all models failed for tier {task!r}: {last_err}")


# == Helper Functions =========================================================


def _is_rate_limit(e: Exception) -> bool:
    """True for 429 / rate-limit errors (across litellm and raw provider errors)."""
    if isinstance(e, getattr(litellm, "RateLimitError", ())):
        return True
    if getattr(e, "status_code", None) == 429:
        return True
    return "rate limit" in str(e).lower()
