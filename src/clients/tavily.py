# src/clients/tavily.py
"""Tavily search transport: the pages answering one query, with their extracted text.

One request per query. Results come back best-scoring first and are flattened to the three
fields a prompt needs, with each page's text capped so one long source cannot crowd out the rest.
"""

import logging
import os
import time

import httpx

from src import config
from src.core.errors import AuthError, ExternalError

logger = logging.getLogger(__name__)

SOURCE = "tavily"

SEARCH_URL = "https://api.tavily.com/search"

_MAX_ERROR_BODY_CHARS = 300
_TIMEOUT = httpx.Timeout(config.TAVILY_TIMEOUT_S, connect=config.TAVILY_CONNECT_TIMEOUT_S)


# == Primitive ================================================================


def search(query: str) -> list[dict[str, str]]:
    """The result pages for `query`, best-scoring first, each as {title, url, text}.

    A timeout, a transport error or a 5xx is retried with capped exponential backoff. A 429 is
    not: Tavily returns it for a spent credit balance, which will not recover today, so retrying
    it would only burn the run's clock. A missing key or a 401 raises AuthError, any other failure
    ExternalError; the raise is the terminal signal, so no failure is logged here. Results whose
    text came back empty are dropped — a title and a URL alone are nothing to build on."""
    key = os.environ.get(config.TAVILY_KEY_LABEL)
    if not key:
        raise AuthError(SOURCE, detail=f"{config.TAVILY_KEY_LABEL} unset")
    attempts = max(config.TAVILY_RETRIES, 1)
    backoff = config.BACKOFF_START_S
    for attempt in range(attempts):
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
                timeout=_TIMEOUT,
            )
        except httpx.HTTPError as e:
            error = ExternalError(SOURCE, cause=e)
            transient = True
        else:
            if response.status_code == config.UNAUTHORIZED_STATUS:
                raise AuthError(SOURCE, detail=response.text[:_MAX_ERROR_BODY_CHARS])
            if response.status_code < config.ERROR_STATUS_FLOOR:
                return [page for page in map(_page, _results(response)) if page["text"]]
            error = ExternalError(
                SOURCE, detail=f"{response.status_code}: {response.text[:_MAX_ERROR_BODY_CHARS]}"
            )
            transient = response.status_code >= config.SERVER_ERROR_STATUS
        if not transient or attempt == attempts - 1:
            raise error from error.cause
        logger.warning("⚠️ tavily search failed (%s); backoff %ss", error, backoff)
        time.sleep(backoff)
        backoff = min(backoff * 2, config.BACKOFF_CAP_S)
    raise ExternalError(SOURCE, detail="search retries exhausted")


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
    snippet."""
    text = result.get("raw_content") or result.get("content") or ""
    return {
        "title": str(result.get("title") or ""),
        "url": str(result.get("url") or ""),
        "text": str(text).strip()[: config.TAVILY_MAX_PAGE_CHARS],
    }
