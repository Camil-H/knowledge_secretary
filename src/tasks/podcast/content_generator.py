# src/tasks/podcast/content_generator.py
"""Grounded topic research via Gemini's Google Search tool.

Replaces asking a model to recall source URLs: the search runs inside the model, so the
overview is grounded in pages that exist rather than in plausible-looking identifiers that
resolve to unrelated articles."""

import logging
from pathlib import Path

from google.genai import types

from src.core import gemini
from src.core import ledger as ledger_mod
from src.core.errors import AuthError, ExternalError

logger = logging.getLogger(__name__)

PROMPT = (Path(__file__).parent / "research_prompt.md").read_text()
_MAX_LOGGED_SOURCES = 10


# == Research =================================================================


def research(topic: str) -> str:
    """A search-grounded overview of the topic from the first grounding-capable model that
    answers; "" when every candidate is spent, failing or empty.

    Only models flagged `search` are candidates — the rest cannot run the grounding tool, and
    an ungrounded overview defeats the point of this step. Runs on the shared Gemini primitive,
    so the day's ledger, pacing and error typing are the same as general text. AuthError
    propagates: a credential failure is not something a later candidate recovers from."""
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction=PROMPT,
    )
    ledger = ledger_mod.load()
    for model in [m for m in gemini.TEXT_MODELS if m.search]:
        if not ledger_mod.available(ledger, model.id, model.rpd):
            continue
        try:
            response = gemini.generate(model, topic, config, ledger=ledger)
        except AuthError:
            raise
        except ExternalError as e:
            logger.warning("⚠️ research model=%s unavailable, next candidate: %s", model.id, e)
            continue
        _log_sources(response)
        text = response.text
        if text and text.strip():
            return text
        logger.warning("⚠️ research model=%s returned empty, next candidate", model.id)
    return ""


# == Helper Functions =========================================================


def _log_sources(response: types.GenerateContentResponse) -> None:
    """Record which pages the answer was grounded in, so a bad episode can be traced back."""
    for candidate in response.candidates or []:
        metadata = candidate.grounding_metadata
        chunks = (metadata.grounding_chunks or []) if metadata else []
        sources = [c.web.uri for c in chunks if c.web and c.web.uri]
        if sources:
            logger.info(
                "research: grounded in %d source(s): %s",
                len(sources),
                ", ".join(sources[:_MAX_LOGGED_SOURCES]),
            )
