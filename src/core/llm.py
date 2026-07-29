# src/core/llm.py
"""LLM transport: the Google AI Studio free tier first, OpenRouter's free models behind it.

The two tiers are separate credentials on purpose — a broken Google key degrades to
OpenRouter instead of taking the newsletter, the youtube digest and the podcast down at once.
Google's free tier has no quota API, so a per-day ledger (src/core/ledger.py) plus 429s are
the only truth about what is left.
"""

import logging

from src.core import gemini, openrouter
from src.core import ledger as ledger_mod
from src.core.errors import AuthError

logger = logging.getLogger(__name__)


# == Cascade ==================================================================


def call(system: str, user: str, *, max_tokens: int | None = None) -> str:
    """First non-empty completion from the Google tier, then the OpenRouter tier.

    All tiers dry raises ExternalError. An OpenRouter auth failure propagates as AuthError;
    a Google one does not, because OpenRouter is an independent credential."""
    ledger = ledger_mod.load()
    try:
        text = gemini.call(system, user, max_tokens, ledger=ledger)
    except AuthError as e:
        # deliberate cross-tier degradation: one bad Google key must not down every product
        logger.warning("⚠️ llm google tier unavailable, degrading to openrouter: %s", e)
        text = ""
    return text or openrouter.call(system, user, max_tokens)
