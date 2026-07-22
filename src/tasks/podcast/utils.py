"""URL-reachability helpers for the podcast's source discovery."""

import asyncio

import httpx

_URL_CHECK_TIMEOUT_S = 10


async def validate_urls(urls: list[str]) -> list[str]:
    """Keep only the URLs that respond < 400, checked concurrently."""
    if not urls:
        return []
    async with httpx.AsyncClient(timeout=_URL_CHECK_TIMEOUT_S, follow_redirects=True) as client:
        oks = await asyncio.gather(*(_url_ok(client, url) for url in urls))
    return [url for url, ok in zip(urls, oks, strict=True) if ok]


async def _url_ok(client: httpx.AsyncClient, url: str) -> bool:
    try:
        resp = await client.head(url)
        if resp.status_code >= 400:  # some servers reject HEAD — confirm with GET
            resp = await client.get(url)
        return resp.status_code < 400
    except Exception:
        return False
