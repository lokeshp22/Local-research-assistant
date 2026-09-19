"""Thin async Tavily wrapper (search + extract).

Deliberately raw httpx rather than the official SDK: the SDK's async surface
lags its sync one, and we want one retry/backoff policy shared with the Exa
client. The API key is read from settings, which reads it from the environment.
It is never logged.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from tools.budget import Budget
from tools.parsing import normalize_url, truncate

SEARCH_URL = "https://api.tavily.com/search"
EXTRACT_URL = "https://api.tavily.com/extract"

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3


class TavilyError(RuntimeError):
    """Raised after retries are exhausted. Callers degrade; they do not crash."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class TavilyClient:
    def __init__(self, settings, budget: Budget | None = None):
        self.s = settings
        self.budget = budget or Budget()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.call_timeout_s, connect=10.0),
            headers={
                "Authorization": f"Bearer {settings.tavily_api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "TavilyClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # -- public ------------------------------------------------------------
    async def search(self, query: str, max_results: int | None = None) -> list[dict]:
        max_results = max_results or self.s.tavily_max_results
        depth = self.s.tavily_search_depth
        payload = {
            "query": truncate(query, 400),
            "max_results": max_results,
            "search_depth": depth,
            "include_answer": False,
            "include_raw_content": False,
        }
        data = await self._post(SEARCH_URL, payload, label="search")
        self.budget.record_tavily_search(depth)
        return self._parse_search(data)

    async def extract(self, urls: list[str]) -> list[dict]:
        clean = []
        for url in urls:
            normalized = normalize_url(url)
            if normalized and normalized not in clean:
                clean.append(normalized)
        clean = clean[: self.s.extract_top_n]
        if not clean:
            return []
        payload = {"urls": clean, "extract_depth": "basic"}
        data = await self._post(EXTRACT_URL, payload, label="extract")
        self.budget.record_tavily_extract(len(clean), "basic")
        return self._parse_extract(data)

    # -- internals ---------------------------------------------------------
    async def _post(self, url: str, payload: dict, *, label: str) -> dict:
        last: TavilyError | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = await self._client.post(url, json=payload)
            except httpx.TimeoutException as exc:
                last = TavilyError(f"Tavily {label} timed out: {exc}", retryable=True)
            except httpx.HTTPError as exc:
                last = TavilyError(f"Tavily {label} transport error: {exc}", retryable=True)
            else:
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                    except ValueError as exc:
                        raise TavilyError(f"Tavily {label} returned non-JSON: {exc}") from exc
                    return body if isinstance(body, dict) else {"results": body}
                retryable = resp.status_code in RETRYABLE_STATUS
                last = TavilyError(
                    f"Tavily {label} HTTP {resp.status_code}: {truncate(resp.text, 200)}",
                    status=resp.status_code,
                    retryable=retryable,
                )
                if not retryable:
                    raise last
                if resp.status_code == 429:
                    await asyncio.sleep(_retry_after(resp, attempt))
                    continue
            if attempt < MAX_ATTEMPTS - 1:
                await asyncio.sleep(0.6 * (2**attempt))
        raise last or TavilyError(f"Tavily {label} failed")

    @staticmethod
    def _parse_search(data: Any) -> list[dict]:
        raw = data.get("results") if isinstance(data, dict) else data
        out: list[dict] = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            url = normalize_url(item.get("url"))
            if not url:
                continue
            out.append(
                {
                    "url": url,
                    "title": truncate(item.get("title") or url, 200),
                    "snippet": truncate(item.get("content") or "", 900),
                    "score": item.get("score"),
                }
            )
        return out

    @staticmethod
    def _parse_extract(data: Any) -> list[dict]:
        out: list[dict] = []
        if not isinstance(data, dict):
            return out
        for item in data.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = normalize_url(item.get("url"))
            content = item.get("raw_content") or item.get("content") or ""
            if not url or not content:
                continue
            out.append({"url": url, "text": truncate(content, 6000)})
        for item in data.get("failed_results") or []:
            if isinstance(item, dict) and item.get("url"):
                out.append(
                    {
                        "url": normalize_url(item.get("url")),
                        "text": "",
                        "error": truncate(item.get("error") or "extraction failed", 200),
                    }
                )
        return out


def _retry_after(resp: httpx.Response, attempt: int) -> float:
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return min(float(header), 20.0)
        except ValueError:
            pass
    return 1.0 * (2**attempt)
