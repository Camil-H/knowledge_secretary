"""Article-body extraction from a URL (trafilatura). Degrades to None."""

import logging

import trafilatura

logger = logging.getLogger(__name__)


def article_text(url: str) -> str | None:
    """Return the extracted main article text, or None if unavailable."""
    try:
        downloaded = trafilatura.fetch_url(url)
        return trafilatura.extract(downloaded) if downloaded else None
    except Exception as e:
        logger.warning("⚠️ url %s failed: %s", url, e)
        return None
