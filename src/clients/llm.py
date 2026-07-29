# src/clients/llm.py
"""LLM transport: the Google AI Studio free tier first, OpenRouter's free models behind it.

The two tiers are separate credentials on purpose — a broken Google key degrades to
OpenRouter instead of taking the newsletter, the youtube digest and the podcast down at once.
Google's free tier has no quota API, so what is left of it is metered locally
(src/core/ledger.py).
"""

import logging

from src.clients import gemini, openrouter
from src.core import ledger as ledger_mod

logger = logging.getLogger(__name__)


# == Cascade ==================================================================


def call(system: str, user: str, *, max_tokens: int | None = None) -> str:
    """First non-empty completion from the Google tier, then the OpenRouter tier.

    Any Google-side failure degrades to OpenRouter, which is an independent credential.
    OpenRouter's own failures propagate — it is the last tier, so there is nothing left to
    fall back to and the caller decides what a dry cascade means."""
    ledger = ledger_mod.load()
    try:
        text = gemini.call(system, user, max_tokens, ledger=ledger)
    except Exception as e:
        logger.warning("⚠️ llm google tier unavailable, degrading to openrouter: %s", e)
        text = ""
    return text or openrouter.call(system, user, max_tokens)
