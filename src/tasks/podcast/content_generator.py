# src/tasks/podcast/content_generator.py
"""Grounded topic research via Gemini's Google Search tool.

Replaces asking a model to recall source URLs: the search runs inside the model, so the
overview is grounded in pages that exist rather than in plausible-looking identifiers that
resolve to unrelated articles."""

import logging
from pathlib import Path

from src.clients import gemini
from src.core import ledger as ledger_mod

logger = logging.getLogger(__name__)

PROMPT = (Path(__file__).parent / "research_prompt.md").read_text()


# == Research =================================================================


def research(topic: str) -> str:
    """A search-grounded overview of the topic from the first grounding-capable model that
    answers; "" when every candidate is spent, failing or empty.

    `search=True` is the whole request: it both restricts the cascade to models that can run the
    grounding tool and attaches it. The tier cascade, ledger, pacing and error typing are the
    shared Gemini ones, so AuthError propagates here too: a credential failure is not something
    a later candidate recovers from."""
    return gemini.call(PROMPT, topic, ledger=ledger_mod.load(), search=True)
