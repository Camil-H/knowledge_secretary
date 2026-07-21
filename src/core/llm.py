"""Per-task tiered LLM calls with dynamic zero-cost fallback.

Primaries come from config (models.<tier>.primary). If models.<tier>.openrouter_free
is true, we append free OpenRouter models fetched live and ranked per tier:
  - "podcast"  -> rank by max_completion_tokens (needs long OUTPUT)
  - otherwise  -> rank by context_length
LiteLLM reads provider keys from standard env vars (GEMINI_API_KEY, GROQ_API_KEY,
OPENROUTER_API_KEY, ...). Candidates are tried in order, falling through on any error.
"""

import logging

import httpx
import litellm

logger = logging.getLogger(__name__)

litellm.drop_params = True  # tolerate provider-specific unsupported params

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
RANK_OUTPUT = "max_output_tokens"
RANK_CONTEXT = "context"
_FREE_LIMIT = 5
_HTTP_TIMEOUT_S = 20


# == Model resolution =========================================================


def _free_openrouter_models(rank: str, *, limit: int = _FREE_LIMIT) -> list[str]:
    """Live-fetch zero-cost OpenRouter models, ranked for the tier.

    Degrades to [] (a missing free list is non-fatal — configured primaries still
    apply), so this is logged as a warning rather than raised.
    """
    try:
        data = httpx.get(OPENROUTER_MODELS_URL, timeout=_HTTP_TIMEOUT_S).json()["data"]
    except Exception as e:
        logger.warning("⚠️ openrouter model list unavailable, using primaries only: %s", e)
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


def resolve_models(task: str, cfg: dict) -> list[str]:
    """Ordered model ids for a tier: configured primaries, then ranked free fallback."""
    tier = cfg["models"][task]
    models = list(tier.get("primary", []))
    if tier.get("openrouter_free"):
        rank = RANK_OUTPUT if task == "podcast" else RANK_CONTEXT
        models += _free_openrouter_models(rank)
    return models


# == Completion ===============================================================


def call(task: str, system: str, user: str, cfg: dict, *, max_tokens: int | None = None) -> str:
    """Try each resolved model in order; return the first non-empty completion.

    Raises RuntimeError only if every candidate fails; the caller decides whether
    to tolerate that and logs the outcome (this primitive logs only in-flight
    fall-through).
    """
    models = resolve_models(task, cfg)
    if not models:
        raise RuntimeError(f"no models resolved for tier {task!r}")

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Exception | None = None
    for model in models:
        try:
            resp = litellm.completion(model=model, messages=messages, max_tokens=max_tokens)
            content = resp.choices[0].message.content
            if content and content.strip():
                return content
            last_err = RuntimeError(f"{model} returned empty content")
        except Exception as e:  # rate limit / auth / 5xx / timeout -> next candidate
            last_err = e
        logger.warning("⚠️ llm tier=%s model=%s unavailable, falling through", task, model)

    raise RuntimeError(f"all models failed for tier {task!r}: {last_err}")
