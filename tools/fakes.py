"""Fixture and offline clients.

Two distinct uses, kept in one module because they share plumbing:

1. `FixtureTavilyClient` / `FixtureExaClient` — used by the *real* contradiction
   tests. They return deliberately conflicting source snippets instead of
   hitting the network, so "did the critic flag the contradiction?" is a
   controlled experiment with a known right answer rather than an LLM opinion.

2. `FakeOllama` — a scripted stand-in for the model, used only by `selftest.py`
   and `run_eval.py --fake`. It exercises every branch of the pipeline (tool
   loop, defensive parsing, dedup, verdicts, synthesis, logging) on a machine
   with no GPU. It is a plumbing harness, NOT a quality signal: never read an
   eval score produced in --fake mode as a claim about the real system.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from typing import Any

from tools.budget import Budget
from tools.parsing import normalize_url

# ---------------------------------------------------------------- fixtures


class FixtureTavilyClient:
    """Returns injected sources for every query. Used by contradiction tests."""

    def __init__(self, sources: list[dict], budget: Budget | None = None, settings=None):
        self.sources = [
            {
                "url": normalize_url(s["url"]),
                "title": s.get("title") or s["url"],
                "snippet": s.get("text") or s.get("snippet") or "",
                "text": s.get("text") or s.get("snippet") or "",
                "score": 1.0,
            }
            for s in sources
        ]
        self.budget = budget or Budget()

    async def search(self, query: str, max_results: int | None = None) -> list[dict]:
        await asyncio.sleep(0)
        self.budget.record_tavily_search("basic")
        return [
            {"url": s["url"], "title": s["title"], "snippet": s["snippet"], "score": s["score"]}
            for s in self.sources[: max_results or len(self.sources)]
        ]

    async def extract(self, urls: list[str]) -> list[dict]:
        await asyncio.sleep(0)
        wanted = {normalize_url(u) for u in urls}
        hits = [s for s in self.sources if s["url"] in wanted] or self.sources
        self.budget.record_tavily_extract(len(hits), "basic")
        return [{"url": s["url"], "text": s["text"]} for s in hits]

    async def aclose(self) -> None:
        return None


class FixtureExaClient:
    """Cross-check sources for the contradiction tests. Returns the *other*
    injected sources, so the critic sees genuine disagreement."""

    def __init__(self, sources: list[dict], budget: Budget | None = None, settings=None):
        self.sources = [
            {
                "url": normalize_url(s["url"]),
                "title": s.get("title") or s["url"],
                "text": s.get("text") or s.get("snippet") or "",
                "author": s.get("author", ""),
                "published_date": s.get("published_date", ""),
            }
            for s in sources
        ]
        self.budget = budget or Budget()

    async def search(self, query, num_results=None, exclude_domains=None) -> list[dict]:
        await asyncio.sleep(0)
        excluded = set(exclude_domains or [])
        out = [s for s in self.sources if not any(d and d in s["url"] for d in excluded)]
        out = out[: num_results or 4]
        self.budget.record_exa_search(len(out), with_text=True)
        return out

    async def aclose(self) -> None:
        return None


# ------------------------------------------------------------- fake search


def _stable_domains(query: str, n: int) -> list[str]:
    pool = [
        "nature.com", "ourworldindata.org", "iea.org", "nber.org", "acm.org",
        "nist.gov", "who.int", "economist.com", "arxiv.org", "reuters.com",
        "reddit.com", "medium.com",
    ]
    seed = int(hashlib.sha256(query.encode()).hexdigest()[:8], 16)
    return [pool[(seed + i * 3) % len(pool)] for i in range(n)]


class FakeTavily:
    def __init__(self, settings=None, budget: Budget | None = None):
        self.budget = budget or Budget()
        self.settings = settings

    async def search(self, query: str, max_results: int | None = None) -> list[dict]:
        await asyncio.sleep(0.01)
        self.budget.record_tavily_search("basic")
        n = max_results or 4
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:48] or "topic"
        return [
            {
                "url": f"https://{d}/{slug}-{i}",
                "title": f"{d}: {query[:60]}",
                "snippet": f"Reporting on {query[:80]}. Figure {i + 1} is cited by several outlets.",
                "score": 0.9 - 0.1 * i,
            }
            for i, d in enumerate(_stable_domains(query, n))
        ]

    async def extract(self, urls: list[str]) -> list[dict]:
        await asyncio.sleep(0.01)
        urls = [normalize_url(u) for u in urls if normalize_url(u)]
        self.budget.record_tavily_extract(len(urls), "basic")
        return [
            {"url": u, "text": f"Full text for {u}. It states the key figures and names the author."}
            for u in urls
        ]

    async def aclose(self) -> None:
        return None


class FakeExa:
    def __init__(self, settings=None, budget: Budget | None = None):
        self.budget = budget or Budget()
        self.settings = settings

    async def search(self, query, num_results=None, exclude_domains=None) -> list[dict]:
        await asyncio.sleep(0.01)
        n = num_results or 3
        out = [
            {
                "url": f"https://{d}/cross-check",
                "title": f"{d} on: {str(query)[:50]}",
                "text": f"Independent coverage broadly agreeing with: {str(query)[:120]}",
                "author": "" if d in {"reddit.com", "medium.com"} else "Staff Reporter",
                "published_date": "2025-02-01",
            }
            for d in _stable_domains("xc:" + str(query), n)
        ]
        self.budget.record_exa_search(len(out), with_text=True)
        return out

    async def aclose(self) -> None:
        return None


# -------------------------------------------------------------- fake model

_ANTONYMS = [
    ("increase", "decrease"), ("increased", "decreased"), ("rose", "fell"),
    ("rising", "falling"), ("higher", "lower"), ("more", "less"),
    ("safe", "unsafe"), ("effective", "ineffective"), ("supports", "refutes"),
    ("grew", "shrank"), ("gain", "loss"), ("positive", "negative"),
    ("accelerating", "slowing"), ("improves", "worsens"),
]
_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%|percent|gw|tw|bn|billion|million|x|years?)", re.I)
_WORD_RE = re.compile(r"[a-z][a-z'-]+")
_STOP = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "what", "how", "why", "does", "did", "has", "have", "its", "their", "been",
    "into", "than", "then", "they", "them", "there", "which", "about", "over",
}


class FakeOllama:
    """Scripted model. Routes on the system prompt, so it stays in sync with the
    real agents: if a prompt changes, the branch that answers it changes too."""

    def __init__(self, settings, budget: Budget | None = None, *, sub_questions: int = 4):
        self.s = settings
        self.budget = budget or Budget()
        self.n_sub = sub_questions
        self.chat_calls = 0

    async def aclose(self) -> None:
        return None

    async def preflight(self) -> None:
        return None

    async def available_models(self) -> list[str]:
        return [self.s.ollama_model, self.s.embed_model]

    # -- embeddings: real cosine over a hashed bag of words ---------------
    async def embed(self, texts: list[str], timeout: float | None = None) -> list[list[float]]:
        await asyncio.sleep(0)
        self.budget.record_ollama(1.0, embed=True)
        dim = 256
        out = []
        for text in texts:
            vec = [0.0] * dim
            words = [w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2]
            for word in words:
                idx = int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    # -- chat -------------------------------------------------------------
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        retries: int = 1,
    ) -> dict:
        started = time.monotonic()
        await asyncio.sleep(0.01)
        self.chat_calls += 1
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        tool_names = {t["function"]["name"] for t in (tools or [])}

        try:
            if "emit_subquestions" in tool_names:
                return self._plan(user)
            if "tavily_search" in tool_names:
                return self._search_turn(messages, user)
            if "fact-checking critic" in system:
                return self._verdict(user)
            if "different search workers" in system:
                return self._contradiction(user)
            if "final research report" in system:
                return self._synthesize(user)
            if "JUDGE" in system or "rubric" in system.lower():
                return self._judge(user)
            if "baseline" in system.lower():
                return self._synthesize(user)
            return _msg(_json_text({"ok": True}))
        finally:
            self.budget.record_ollama((time.monotonic() - started) * 1000)

    # -- branches ---------------------------------------------------------
    def _plan(self, query: str) -> dict:
        query = query.strip().rstrip("?")
        base = [
            f"What is the current state of {query}, with figures from the last two years?",
            f"What mechanisms or causes explain {query}?",
            f"What criticisms or competing interpretations exist regarding {query}?",
            f"What practical consequences follow from {query}?",
            f"What criticisms or competing interpretations exist about {query}?",  # near-dup of #3
        ]
        return _msg("", [{"name": "emit_subquestions", "arguments": {"sub_questions": base[: self.n_sub]}}])

    def _search_turn(self, messages: list[dict], user: str) -> dict:
        import json as _json

        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        sub = user.replace("Sub-question:", "").strip()
        if not tool_msgs:
            # Arguments as a JSON *string* on purpose — exercises coerce_tool_args.
            return _msg("", [{"name": "tavily_search", "arguments": f'{{"query": "{sub[:120]}"}}'}])

        # Read what the tools actually returned. A real model derives its claims
        # from the retrieved text; so must this one, or injected source conflicts
        # would never reach the critic and the contradiction tests would be vacuous.
        docs: dict[str, str] = {}
        for msg in tool_msgs:
            try:
                payload = _json.loads(str(msg.get("content", "")))
            except (ValueError, TypeError):
                continue
            for item in payload.get("results") or []:
                if not isinstance(item, dict) or not item.get("url"):
                    continue
                body = (item.get("text") or item.get("snippet") or "").strip()
                if body and len(body) > len(docs.get(item["url"], "")):
                    docs[item["url"]] = body
                docs.setdefault(item["url"], body)

        urls = list(docs.keys())
        if len(tool_msgs) == 1:
            return _msg("", [{"name": "tavily_extract", "arguments": {"urls": urls[:3]}}])

        claims = []
        for i, u in enumerate(urls[:3]):
            body = docs.get(u, "")
            text = body[:300].strip() or f"Relevant to {sub[:110]}"
            claims.append(
                {
                    "text": text,
                    "source_url": u,
                    "confidence": "high" if i == 0 else "medium",
                    "confidence_note": f"stated in the extracted text of {normalize_url(u).split('//')[-1].split('/')[0]}",
                }
            )
        if urls:
            # One deliberately unverifiable citation per sub-question, so the
            # critic's URL check has something to catch in offline runs.
            claims.append(
                {
                    "text": f"A secondary account suggests additional context for {sub[:90]}",
                    "source_url": "https://example.invalid/not-retrieved",
                    "confidence": "low",
                    "confidence_note": "recalled, not retrieved",
                }
            )
        return _msg("", [{"name": "emit_claims", "arguments": {"claims": claims}}])

    def _verdict(self, user: str) -> dict:
        retrieved = "RETRIEVED BY THE WORKER: yes" in user
        low_quality = any(d in user for d in ("reddit.com", "medium.com", "quora.com"))
        return _msg(
            _json_text(
                {
                    "support": "supported" if retrieved else "unsupported",
                    "corroboration": "corroborated" if retrieved else "absent",
                    "source_quality": "low" if low_quality else "ok",
                    "reason": ""
                    if retrieved and not low_quality
                    else ("no identifiable author or publication" if low_quality else "cited page was never retrieved"),
                }
            )
        )

    def _contradiction(self, user: str) -> dict:
        parts = user.split("CLAIM B")
        a, b = (parts[0].lower(), parts[1].lower()) if len(parts) == 2 else (user.lower(), "")
        if _polarity_conflict(a, b) or _numeric_conflict(a, b):
            return _msg(
                _json_text(
                    {"contradiction": True, "explanation": "the two sources give opposing values for the same quantity"}
                )
            )
        return _msg(_json_text({"contradiction": False, "explanation": "compatible"}))

    def _synthesize(self, user: str) -> dict:
        question = user.split("\n")[0].replace("RESEARCH QUESTION:", "").strip()
        cites = sorted({int(n) for n in re.findall(r"^\[(\d+)\]", user, re.M)})
        body = [f"## {question or 'Findings'}", ""]
        if cites:
            joined = "".join(f"[{c}]" for c in cites[:3])
            body.append(
                f"The retrieved sources converge on a consistent account of the question {joined}. "
                f"The strongest evidence comes from the primary reporting {'[%d]' % cites[0]}."
            )
            body.append("")
            body.append("### Detail")
            for cite in cites:
                body.append(f"- Supporting evidence recorded under citation [{cite}].")
        else:
            body.append("No verified claims were available for synthesis.")
        return _msg("\n".join(body))

    def _judge(self, user: str) -> dict:
        length = len(user)
        cites = len(set(re.findall(r"\[(\d+)\]", user)))
        depth = max(1, min(5, 1 + cites // 2 + (1 if length > 1200 else 0)))
        return _msg(
            _json_text(
                {
                    "depth": depth,
                    "citation_quality": max(1, min(5, cites)),
                    "reason": "scored from citation count and length (offline stub)",
                }
            )
        )


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _polarity_conflict(a: str, b: str) -> bool:
    return any(
        (_has_word(a, p) and _has_word(b, q)) or (_has_word(a, q) and _has_word(b, p))
        for p, q in _ANTONYMS
    )


def _numeric_conflict(a: str, b: str) -> bool:
    na = {(m.group(2).lower(), float(m.group(1))) for m in _NUM_RE.finditer(a)}
    nb = {(m.group(2).lower(), float(m.group(1))) for m in _NUM_RE.finditer(b)}
    units_a = {u for u, _ in na}
    for unit, value in nb:
        if unit in units_a and all(abs(value - v) > 1e-9 for u, v in na if u == unit):
            return True
    return False


def _msg(content: str, tool_calls: list[dict] | None = None) -> dict:
    from tools.parsing import coerce_tool_args

    return {
        "content": content,
        "tool_calls": [
            {"name": c["name"], "arguments": coerce_tool_args(c["arguments"])}
            for c in (tool_calls or [])
        ],
        "done_reason": "stop",
    }


def _json_text(obj: Any) -> str:
    import json

    return json.dumps(obj)
