"""Thin async wrapper around the Ollama HTTP API.

The important part is the semaphore. Ollama serves `OLLAMA_NUM_PARALLEL`
requests concurrently and queues the rest; it does not do continuous batching,
so firing more than that in parallel buys nothing and makes every in-flight
request slower. A single semaphore sized to num_parallel is shared by the
planner, both search workers, and the critic, so the *whole process* never has
more than two generations in flight regardless of which agent wants one.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from tools.budget import Budget
from tools.parsing import coerce_tool_args, strip_think


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, settings, budget: Budget | None = None):
        self.s = settings
        self.budget = budget or Budget()
        self._sem = asyncio.Semaphore(settings.search_workers)
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_host.rstrip("/"),
            timeout=httpx.Timeout(settings.call_timeout_s * 4, connect=10.0),
        )
        self._supports_think = True
        self._embed_endpoint: str | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # -- health ------------------------------------------------------------
    async def available_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/tags", timeout=10.0)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except (httpx.HTTPError, ValueError) as exc:
            raise OllamaError(f"Ollama not reachable at {self.s.ollama_host}: {exc}") from exc

    async def preflight(self) -> None:
        """Fail early and legibly rather than 20 minutes into an eval run."""
        models = await self.available_models()
        names = {m.split(":")[0] for m in models} | set(models)
        for required in (self.s.ollama_model, self.s.embed_model):
            if required not in models and required.split(":")[0] not in names:
                raise OllamaError(
                    f"Model {required!r} is not pulled. Run: ollama pull {required}"
                )

    # -- chat --------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        retries: int = 1,
    ) -> dict:
        """Returns the assistant message dict with `content` and, when the
        model called a tool, a normalized `tool_calls` list where every
        `arguments` value is guaranteed to be a dict."""
        timeout = timeout or self.s.call_timeout_s
        payload: dict[str, Any] = {
            "model": self.s.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.s.temperature if temperature is None else temperature,
                "num_ctx": self.s.num_ctx,
            },
        }
        if tools:
            payload["tools"] = tools
        if json_mode and not tools:
            payload["format"] = "json"
        if self._supports_think:
            payload["think"] = self.s.think

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            started = time.monotonic()
            try:
                async with self._sem:
                    resp = await asyncio.wait_for(
                        self._client.post("/api/chat", json=payload), timeout=timeout
                    )
                if resp.status_code == 400 and self._supports_think and "think" in payload:
                    # Older Ollama builds reject the `think` field outright.
                    self._supports_think = False
                    payload.pop("think", None)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except asyncio.TimeoutError as exc:
                last_exc = OllamaError(f"Ollama chat timed out after {timeout}s")
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_exc = OllamaError(f"Ollama chat failed: {exc}")
            else:
                self.budget.record_ollama((time.monotonic() - started) * 1000)
                return self._normalize_message(data)
            self.budget.record_ollama((time.monotonic() - started) * 1000)
            if attempt < retries:
                await asyncio.sleep(0.8 * (attempt + 1))
        raise last_exc or OllamaError("Ollama chat failed")

    @staticmethod
    def _normalize_message(data: dict) -> dict:
        message = data.get("message") or {}
        content = strip_think(message.get("content") or "")
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name") or call.get("name") or ""
            if not name:
                continue
            calls.append({"name": str(name), "arguments": coerce_tool_args(fn.get("arguments"))})
        return {
            "content": content,
            "tool_calls": calls,
            "done_reason": data.get("done_reason"),
            "eval_count": data.get("eval_count"),
            "prompt_eval_count": data.get("prompt_eval_count"),
        }

    # -- embeddings --------------------------------------------------------
    async def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]:
        """nomic-embed-text over the shared semaphore. CPU is fine here, but it
        still goes through Ollama, so it still occupies a slot."""
        texts = [t for t in texts]
        if not texts:
            return []
        timeout = timeout or self.s.call_timeout_s
        started = time.monotonic()
        try:
            async with self._sem:
                vectors = await asyncio.wait_for(self._embed_request(texts), timeout=timeout)
        except asyncio.TimeoutError as exc:
            self.budget.record_ollama((time.monotonic() - started) * 1000, embed=True)
            raise OllamaError(f"Ollama embed timed out after {timeout}s") from exc
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self.budget.record_ollama((time.monotonic() - started) * 1000, embed=True)
            raise OllamaError(f"Ollama embed failed: {exc}") from exc
        self.budget.record_ollama((time.monotonic() - started) * 1000, embed=True)
        return vectors

    async def _embed_request(self, texts: list[str]) -> list[list[float]]:
        if self._embed_endpoint != "/api/embeddings":
            try:
                resp = await self._client.post(
                    "/api/embed", json={"model": self.s.embed_model, "input": texts}
                )
                resp.raise_for_status()
                vectors = resp.json().get("embeddings")
                if isinstance(vectors, list) and vectors:
                    self._embed_endpoint = "/api/embed"
                    return [[float(x) for x in v] for v in vectors]
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
            self._embed_endpoint = "/api/embeddings"
        # Legacy endpoint: one prompt per request.
        out: list[list[float]] = []
        for text in texts:
            resp = await self._client.post(
                "/api/embeddings", json={"model": self.s.embed_model, "prompt": text}
            )
            resp.raise_for_status()
            out.append([float(x) for x in resp.json()["embedding"]])
        return out
