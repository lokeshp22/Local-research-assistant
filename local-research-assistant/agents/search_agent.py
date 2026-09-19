"""Search agent — one instance per worker, two workers total.

Handles exactly one sub-question at a time: Tavily search, Tavily extract on
the top results, then a structured list of claims each tagged with the source
URL it came from and a confidence note.

Guardrails that matter here:
  * every model call and every HTTP call is wrapped in a hard timeout;
  * the tool-calling loop is capped at MAX_TURNS so a model that keeps calling
    search forever cannot run up credits;
  * a Tavily failure becomes a tool *result* the model can see and route
    around, not an exception that unwinds the run;
  * every emitted source URL is checked against the set of URLs Tavily actually
    returned. A claim citing a URL the agent never saw is marked
    url_verified=false, which the critic treats as a hard signal.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field

from tools.budget import Budget
from tools.ollama_client import OllamaError
from tools.parsing import extract_json, normalize_url, truncate

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Web search. Returns titles, URLs and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tavily_extract",
            "description": "Fetch the full text of up to 4 URLs returned by tavily_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs copied exactly from tavily_search results.",
                    }
                },
                "required": ["urls"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emit_claims",
            "description": "Finish. Emit the claims you can support from the sources you read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "One factual claim, self-contained, one sentence.",
                                },
                                "source_url": {
                                    "type": "string",
                                    "description": "The exact URL this claim came from.",
                                },
                                "confidence": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                                "confidence_note": {
                                    "type": "string",
                                    "description": "One short line on why that confidence.",
                                },
                            },
                            "required": ["text", "source_url", "confidence"],
                        },
                    }
                },
                "required": ["claims"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a search worker. You are given one sub-question and nothing else.

Procedure:
1. Call tavily_search once with a well-formed query for the sub-question.
2. Call tavily_extract once on the 3-4 most promising URLs from those results.
3. Call emit_claims with the claims you can actually support.

Hard rules:
- source_url must be copied character-for-character from a tavily_search or
  tavily_extract result. Never invent, shorten, or reconstruct a URL.
- One fact per claim. A claim must be checkable against its source on its own.
- If a source does not support something, leave it out. An empty claims list is a
  valid answer and is better than a guessed one.
- Set confidence to low when only one source says it, or when the source hedges.
- You have at most {max_turns} tool calls in total. Call emit_claims before you run out."""


@dataclass
class Claim:
    id: str
    sub_question: str
    text: str
    source_url: str
    confidence: str
    confidence_note: str
    url_verified: bool

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "sub_question": self.sub_question,
            "text": self.text,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "confidence_note": self.confidence_note,
            "url_verified": self.url_verified,
        }


@dataclass
class SubQuestionResult:
    sub_question: str
    status: str = "ok"  # ok | partial | failed
    claims: list[Claim] = field(default_factory=list)
    turns: int = 0
    sources_seen: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sub_question": self.sub_question,
            "status": self.status,
            "turns": self.turns,
            "claim_count": len(self.claims),
            "sources_seen": self.sources_seen,
            "failures": self.failures,
        }


async def run_sub_question(
    sub_question: str, *, ollama, tavily, budget: Budget, settings
) -> SubQuestionResult:
    result = SubQuestionResult(sub_question=sub_question)
    seen: dict[str, str] = {}  # normalized url -> title

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_turns=settings.max_turns)},
        {"role": "user", "content": f"Sub-question: {sub_question}"},
    ]

    searched = False
    for turn in range(settings.max_turns):
        result.turns = turn + 1
        try:
            message = await ollama.chat(
                messages, tools=SEARCH_TOOLS, timeout=settings.call_timeout_s, retries=0
            )
        except OllamaError as exc:
            result.failures.append(
                budget.record_failure("search", "ollama", str(exc), sub_question=sub_question)
            )
            result.status = "failed" if not result.claims else "partial"
            return result

        calls = message.get("tool_calls") or []
        if not calls:
            # No tool call. Try to salvage claims from prose, else nudge once.
            salvaged = _claims_from_text(message.get("content") or "")
            if salvaged:
                result.claims = _build_claims(salvaged, sub_question, seen)
                result.status = "ok" if result.claims else "partial"
                break
            messages.append({"role": "assistant", "content": message.get("content") or ""})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You did not call a tool. Call tavily_search now, or call "
                        "emit_claims if you already have enough."
                        if not searched
                        else "Call emit_claims now with what you have."
                    ),
                }
            )
            continue

        messages.append(
            {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [
                    {"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls
                ],
            }
        )

        finished = False
        for call in calls:
            name, args = call["name"], call["arguments"]

            if name == "emit_claims":
                raw_claims = args.get("claims")
                if isinstance(raw_claims, str):
                    raw_claims = extract_json(raw_claims)
                if isinstance(raw_claims, dict):
                    raw_claims = raw_claims.get("claims")
                built = _build_claims(raw_claims or [], sub_question, seen)
                result.claims = built
                result.status = "ok" if built else "partial"
                finished = True
                break

            if name == "tavily_search":
                query = args.get("query") or args.get("q") or sub_question
                if not isinstance(query, str) or not query.strip():
                    query = sub_question
                payload = await _safe_search(query, tavily, budget, result, sub_question, settings)
                searched = True
                for item in payload.get("results", []):
                    seen[item["url"]] = item.get("title", "")
                messages.append(
                    {"role": "tool", "name": name, "content": json.dumps(payload, ensure_ascii=False)}
                )
                continue

            if name == "tavily_extract":
                urls = args.get("urls") or args.get("url") or []
                if isinstance(urls, str):
                    parsed = extract_json(urls)
                    urls = parsed if isinstance(parsed, list) else [urls]
                urls = [u for u in (normalize_url(u) for u in urls) if u][: settings.extract_top_n]
                if not urls:
                    urls = list(seen.keys())[: settings.extract_top_n]
                payload = await _safe_extract(urls, tavily, budget, result, sub_question, settings)
                for item in payload.get("results", []):
                    seen.setdefault(item["url"], "")
                messages.append(
                    {"role": "tool", "name": name, "content": json.dumps(payload, ensure_ascii=False)}
                )
                continue

            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": json.dumps({"error": f"unknown tool {name!r}"}),
                }
            )

        if finished:
            break
    else:
        # Turn budget exhausted without emit_claims. Spend one final, tool-free
        # call to harvest whatever the agent has rather than discarding the work.
        result.status = "partial"
        try:
            message = await ollama.chat(
                messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "Tool budget exhausted. Reply with JSON only: "
                            '{"claims":[{"text":"...","source_url":"...","confidence":"low",'
                            '"confidence_note":"..."}]}. Use [] if you have nothing supported.'
                        ),
                    }
                ],
                timeout=settings.call_timeout_s,
                json_mode=True,
                retries=0,
            )
            result.claims = _build_claims(
                _claims_from_text(message.get("content") or ""), sub_question, seen
            )
        except OllamaError as exc:
            result.failures.append(
                budget.record_failure("search", "ollama-final", str(exc), sub_question=sub_question)
            )

    result.sources_seen = sorted(seen.keys())
    if not result.claims and result.status == "ok":
        result.status = "partial"
    return result


# -- tool execution with graceful degradation -----------------------------


async def _safe_search(query, tavily, budget, result, sub_question, settings) -> dict:
    try:
        results = await asyncio.wait_for(tavily.search(query), timeout=settings.call_timeout_s)
        return {"results": results}
    except asyncio.TimeoutError:
        detail = f"tavily.search timed out after {settings.call_timeout_s}s"
    except Exception as exc:  # TavilyError and anything the transport throws
        detail = f"tavily.search failed: {exc}"
    result.failures.append(
        budget.record_failure("search", "tavily_search", detail, sub_question=sub_question)
    )
    return {"results": [], "error": detail, "advice": "Continue with what you have, or emit_claims with []."}


async def _safe_extract(urls, tavily, budget, result, sub_question, settings) -> dict:
    if not urls:
        return {"results": [], "error": "no URLs to extract"}
    try:
        results = await asyncio.wait_for(tavily.extract(urls), timeout=settings.call_timeout_s)
        return {"results": [r for r in results if r.get("text")], "failed": [r for r in results if not r.get("text")]}
    except asyncio.TimeoutError:
        detail = f"tavily.extract timed out after {settings.call_timeout_s}s"
    except Exception as exc:
        detail = f"tavily.extract failed: {exc}"
    result.failures.append(
        budget.record_failure("search", "tavily_extract", detail, sub_question=sub_question)
    )
    return {"results": [], "error": detail, "advice": "Fall back to the search snippets you already have."}


# -- claim construction ----------------------------------------------------


def _claims_from_text(text: str) -> list:
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        for key in ("claims", "results", "items"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        return []
    if isinstance(parsed, list):
        return parsed
    return []


def _build_claims(raw_claims, sub_question: str, seen: dict[str, str]) -> list[Claim]:
    claims: list[Claim] = []
    if not isinstance(raw_claims, list):
        return claims
    for item in raw_claims:
        if isinstance(item, str):
            item = {"text": item, "source_url": "", "confidence": "low"}
        if not isinstance(item, dict):
            continue
        text = truncate(item.get("text") or item.get("claim") or "", 600)
        if len(text) < 12:
            continue
        url = normalize_url(item.get("source_url") or item.get("url") or item.get("source") or "")
        confidence = str(item.get("confidence") or "medium").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        note = truncate(item.get("confidence_note") or item.get("note") or "", 300)
        verified = bool(url) and url in seen
        if not verified and not note:
            note = "source URL was not among the results this worker retrieved"
        claims.append(
            Claim(
                id=f"c_{uuid.uuid4().hex[:10]}",
                sub_question=sub_question,
                text=text,
                source_url=url,
                confidence=confidence,
                confidence_note=note,
                url_verified=verified,
            )
        )
    return claims
