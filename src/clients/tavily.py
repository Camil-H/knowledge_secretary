# src/clients/tavily.py
"""Tavily search transport: the pages answering one query, with their extracted text.

Grounding lives in a search API rather than inside a model because no free-tier Gemini model can
run the google_search tool. What matters is preserved either way: the material rests on pages
that exist, instead of on source identifiers a model recalled and largely invented.
"""

import logging
import os

import httpx

from src import config
from src.core.errors import AuthError, ExternalError

logger = logging.getLogger(__name__)

SOURCE = "tavily"

SEARCH_URL = "https://api.tavily.com/search"

_MAX_ERROR_BODY_CHARS = 300


# == Primitive ================================================================


def search(query: str) -> list[dict[str, str]]:
    """The result pages for `query`, best-scoring first, each as {title, url, text}.

    A missing key or a 401 raises AuthError, any other failure ExternalError; the raise is the
    terminal signal, so no failure is logged here. Results whose text came back empty are
    dropped — a title and a URL alone are nothing to build an episode on."""
    key = os.environ.get(config.TAVILY_KEY_LABEL)
    if not key:
        raise AuthError(SOURCE, detail=f"{config.TAVILY_KEY_LABEL} unset")
    logger.info("🚀 tavily search %r", query)
    try:
        response = httpx.post(
            SEARCH_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "query": query,
                "search_depth": config.TAVILY_SEARCH_DEPTH,
                "max_results": config.TAVILY_MAX_RESULTS,
                "include_raw_content": True,
            },
            timeout=config.HTTP_TIMEOUT_S,
        )
    except httpx.HTTPError as e:
        raise ExternalError(SOURCE, cause=e) from e
    if response.status_code == config.UNAUTHORIZED_STATUS:
        raise AuthError(SOURCE, detail=response.text[:_MAX_ERROR_BODY_CHARS])
    if response.status_code >= config.ERROR_STATUS_FLOOR:
        raise ExternalError(
            SOURCE, detail=f"{response.status_code}: {response.text[:_MAX_ERROR_BODY_CHARS]}"
        )
    return [page for page in map(_page, _results(response)) if page["text"]]


# == Helper Functions =========================================================


def _results(response: httpx.Response) -> list[dict]:
    """The response's result list, or [] for any shape that is not one."""
    try:
        body = response.json()
    except ValueError as e:
        raise ExternalError(SOURCE, cause=e) from e
    results = body.get("results") if isinstance(body, dict) else None
    return [r for r in results if isinstance(r, dict)] if isinstance(results, list) else []


def _page(result: dict) -> dict[str, str]:
    """One result flattened to {title, url, text}, its text capped for the prompt.

    raw_content is the full extracted page and is preferred over the answer-shaped `content`
    snippet; whichever is used, one long page must not crowd out every other source."""
    text = result.get("raw_content") or result.get("content") or ""
    return {
        "title": str(result.get("title") or ""),
        "url": str(result.get("url") or ""),
        "text": str(text).strip()[: config.TAVILY_MAX_PAGE_CHARS],
    }
