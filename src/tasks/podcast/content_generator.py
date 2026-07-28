# src/tasks/podcast/content_generator.py
"""Grounded topic research via Gemini's Google Search tool.

Replaces asking a model to recall source URLs: the search runs inside the model, so the
overview is grounded in pages that exist rather than in plausible-looking identifiers that
resolve to unrelated articles."""

import logging
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Search grounding is free for 5k prompts/month on the 3.x family, and Flash is the newest
# tier still on the free plan — Pro models moved behind billing in April 2026.
MODEL = "gemini-3.6-flash"
PROMPT = (Path(__file__).parent / "research_prompt.md").read_text()
_MAX_LOGGED_SOURCES = 10


# == Research =================================================================


def research(topic: str, *, api_key: str) -> str:
    """A search-grounded overview of the topic; "" when the model returns no text.

    Raises whatever the SDK raises — the caller decides whether a failed episode is tolerable."""
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction=PROMPT,
    )
    logger.info("🚀 research model=%s topic=%r", MODEL, topic)
    response = client.models.generate_content(model=MODEL, contents=topic, config=config)
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
