"""Thin async Exa wrapper used only by the critic.

Exa is asked for *independent* corroboration, so we request author and
publication metadata alongside the text — the critic needs it to judge source
quality, which is one of its three required flags.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from tools.budget import Budget
from tools.parsing import normalize_url, truncate

SEARCH_URL = "https://api.exa.ai/search"
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3


class ExaError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class ExaClient:
    def __init__(self, settings, budget: Budget | None = None):
        self.s = settings
        self.budget = budget or Budget()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.call_timeout_s, connect=10.0),
            headers={"x-api-key": settings.exa_api_key, "Content-Type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "ExaClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def search(
        self,
        query: str,
        num_results: int | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[dict]:
        num_results = num_results or self.s.exa_num_results
        payload: dict[str, Any] = {
            "query": truncate(query, 400),
            "numResults": num_results,
            "type": "auto",
            "contents": {"text": {"maxCharacters": 1200}},
        }
        if exclude_domains:
            payload["excludeDomains"] = [d for d in exclude_domains if d][:10]

        last: ExaError | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = await self._client.post(SEARCH_URL, json=payload)
            except httpx.TimeoutException as exc:
                last = ExaError(f"Exa search timed out: {exc}", retryable=True)
            except httpx.HTTPError as exc:
                last = ExaError(f"Exa transport error: {exc}", retryable=True)
            else:
                if resp.status_code == 200:
                    try:
                        body = resp.json()
                    except ValueError as exc:
                        raise ExaError(f"Exa returned non-JSON: {exc}") from exc
                    results = self._parse(body)
                    self.budget.record_exa_search(len(results), with_text=True)
                    return results
                retryable = resp.status_code in RETRYABLE_STATUS
                last = ExaError(
                    f"Exa HTTP {resp.status_code}: {truncate(resp.text, 200)}",
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
        raise last or ExaError("Exa search failed")

    @staticmethod
    def _parse(body: Any) -> list[dict]:
        raw = body.get("results") if isinstance(body, dict) else body
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
                    "text": truncate(item.get("text") or "", 1200),
                    "author": truncate(item.get("author") or "", 120),
                    "published_date": item.get("publishedDate") or "",
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
