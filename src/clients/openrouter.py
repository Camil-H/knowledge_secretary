# src/clients/openrouter.py
"""OpenRouter transport: one chat completion, and the deadline-bounded tier loop over
config.OPENROUTER_MODELS.

Its key is independent of the Google one on purpose, so this tier still answers when the
AI Studio credential is broken.
"""

import logging
import os
import time

import httpx

from src import config
from src.core.errors import AuthError, ExternalError

logger = logging.getLogger(__name__)

SOURCE = "openrouter"
_CASCADE_SOURCE = "llm"

COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_PREFIX = "openrouter/"
_MAX_ERROR_BODY_CHARS = 300


# == Exceptions ===============================================================


class OpenRouterError(Exception):
    """A non-OK OpenRouter response; carries the status and a truncated error body."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        super().__init__(f"openrouter {status_code}: {body}")


# == Tier =====================================================================


def call(system: str, user: str, max_tokens: int | None) -> str:
    """First non-empty OpenRouter completion: a transient status retries the same model with
    backoff, a 401 raises immediately, anything else falls through to the next candidate, and
    all-fail raises ExternalError.
    A wall-clock deadline caps total time so the cascade is abandoned rather than walking
    every model x retry."""
    candidates = config.OPENROUTER_MODELS
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
                status = getattr(e, "status_code", None)
                if status == config.UNAUTHORIZED_STATUS:
                    raise AuthError(SOURCE, cause=e) from e
                retriable = status in config.TRANSIENT_STATUSES
                if retriable and attempt < config.RATE_LIMIT_RETRIES - 1:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break  # outer deadline guard abandons the cascade next iteration
                    logger.warning(
                        "⚠️ llm model=%s refused (%s); backoff %ss", model, status, backoff
                    )
                    time.sleep(min(backoff, remaining))
                    backoff = min(backoff * 2, config.BACKOFF_CAP_S)
                    continue
                logger.warning(
                    "⚠️ llm model=%s unavailable, next candidate: %s status=%s",
                    model,
                    type(e).__name__,
                    status,
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
