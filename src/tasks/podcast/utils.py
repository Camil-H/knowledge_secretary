"""URL-reachability helpers for the podcast's source discovery."""

import asyncio
from urllib.parse import urljoin

import httpx

from src.core.net import is_safe_url

_URL_CHECK_TIMEOUT_S = 10
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


async def reachable_urls(urls: list[str]) -> list[str]:
    """Keep only the URLs that respond < 400, checked concurrently."""
    if not urls:
        return []
    # Redirects are followed manually so each hop can be re-checked against the SSRF guard.
    async with httpx.AsyncClient(timeout=_URL_CHECK_TIMEOUT_S, follow_redirects=False) as client:
        oks = await asyncio.gather(*(_url_ok(client, url) for url in urls))
    return [url for url, ok in zip(urls, oks, strict=True) if ok]


async def _url_ok(client: httpx.AsyncClient, url: str) -> bool:
    try:
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            if not is_safe_url(current):
                return False
            resp = await client.head(current)
            if resp.status_code >= 400:  # some servers reject HEAD — confirm with GET
                resp = await client.get(current)
            location = resp.headers.get("location")
            if resp.status_code in _REDIRECT_STATUSES and location:
                current = urljoin(current, location)
                continue
            return resp.status_code < 400
        return False  # redirect chain too long
    except Exception:
        return False
