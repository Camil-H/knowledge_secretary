"""Article-body extraction: guarded fetch, trafilatura for extraction. Degrades to None."""

import logging

import trafilatura

from src import config
from src.core.url_guard import fetch_following_safe_redirects

logger = logging.getLogger(__name__)


def article_text(url: str) -> str | None:
    """Return the extracted main article text, or None if unavailable."""
    try:
        response = fetch_following_safe_redirects(url)
        if response.status_code >= config.ERROR_STATUS_FLOOR:
            logger.warning("⚠️ url %s degraded: status=%s", url, response.status_code)
            return None
        if not response.text:
            logger.warning("⚠️ url %s degraded: empty body", url)
            return None
        return trafilatura.extract(response.text)
    except Exception as e:  # unsafe hop, transport failure, or trafilatura's assorted parse errors
        logger.warning("⚠️ url %s degraded: %s: %s", url, type(e).__name__, e)
        return None
