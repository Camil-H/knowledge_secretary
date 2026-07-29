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

logger = logging.getLogger(__name__)

PROMPT = (Path(__file__).parent / "research_prompt.md").read_text()
_MAX_LOGGED_SOURCES = 10


# == Research =================================================================


def research(topic: str) -> str:
    """A search-grounded overview of the topic; "" when the model returns no text.

    Runs on the shared Gemini primitive, so it shares the day's ledger, pacing and error
    typing. Raises what the primitive raises — the caller decides whether a failed episode
    is tolerable."""
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction=PROMPT,
    )
    response = gemini.generate(gemini.TEXT_MODELS[0], topic, config, ledger=ledger_mod.load())
    _log_sources(response)
    return response.text or ""


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
