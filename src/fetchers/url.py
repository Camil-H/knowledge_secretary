"""Article-body extraction from a URL (trafilatura). Degrades to None."""

import logging

import trafilatura

from src.core.net import is_safe_url

logger = logging.getLogger(__name__)


def article_text(url: str) -> str | None:
    """Return the extracted main article text, or None if unavailable."""
    if not is_safe_url(url):
        logger.warning("⚠️ url %s degraded: unsafe target", url)
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded) if downloaded else None
    except Exception as e:  # trafilatura raises assorted errors on fetch/extract
        logger.warning("⚠️ url %s degraded: %s", url, type(e).__name__)
        return None
