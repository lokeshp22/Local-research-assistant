"""Critic — runs once, after the search queue drains.

Three jobs, in order:
  1. cross-check each claim against independent sources found via Exa semantic
     search, excluding the claim's own domain so "corroboration" cannot just be
     the same page again;
  2. find contradictions *between* claims produced by different workers;
  3. assign every claim a verdict (kept / flagged / rejected) with a one-line
     reason for every flag and rejection.

The synthesizer is the critic's own final call rather than a fifth model, so
the stage that decided what survives is the stage that writes it up.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from tools.budget import Budget
from tools.embeddings import lexical_overlap, similar_pairs
from tools.ollama_client import OllamaError
from tools.parsing import domain_of, extract_json, truncate

# Domains where "who wrote this" is structurally unanswerable. Used as a prior,
# not a verdict — the model still gets the final say on source quality.
# Contradiction-check tuning.
LEXICAL_OVERLAP_THRESHOLD = 0.5   # share of the shorter claim's content words

LOW_SIGNAL_DOMAINS = {
    "answers.com", "quora.com", "reddit.com", "medium.com", "blogspot.com",
    "wordpress.com", "ezinearticles.com", "scribd.com", "coursehero.com",
    "studocu.com", "chegg.com", "facebook.com", "x.com", "twitter.com",
    "pinterest.com", "tiktok.com",
}

CROSS_CHECK_SYSTEM = """You are a fact-checking critic. You are given one claim, the source it
was drawn from, and independent sources found by a separate semantic search.

Decide three things:
- support: does the claim's own cited source plausibly support it? ("supported",
  "unsupported", or "unknown" if you were given no usable text from it)
- corroboration: do the independent sources agree? ("corroborated",
  "contradicted", "absent")
- source_quality: does the cited source have an identifiable author or
  publication? ("ok", "low", "unknown")

Reply with JSON only, no prose:
{"support":"...","corroboration":"...","source_quality":"...","reason":"one short line"}

The reason must be one line and must name the actual problem if there is one."""

CONTRADICTION_SYSTEM = """You compare research claims that came from different search workers.

Decide whether they genuinely contradict: they cannot both be true as written.
Different emphasis, different scope, different time period, or one being more
specific than the other is NOT a contradiction.

Reply with JSON only:
{"contradiction": true|false, "explanation": "one short line"}"""

SYNTH_SYSTEM = """You are writing the final research report from claims that survived fact-checking.

Rules:
- Use ONLY the numbered claims given. Add no outside knowledge, no hedging filler.
- Cite inline with [1], [2] matching the claim numbers you were given. Every
  substantive sentence carries at least one citation.
- Several claims may support one sentence: write [1][3].
- Organise by theme, not by claim order. Open with a direct answer to the
  question in two or three sentences, then the detail.
- If the claims genuinely conflict or leave a gap, say so plainly in one line.
- Do not write a source list; it is appended for you.
- Plain prose with short markdown headings. No preamble about what you are doing."""


@dataclass
class CriticResult:
    annotated: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)

    @property
    def kept(self) -> list[dict]:
        return [c for c in self.annotated if c["verdict"] == "kept"]

    def counts(self) -> dict:
        out = {"kept": 0, "flagged": 0, "rejected": 0}
        for claim in self.annotated:
            out[claim["verdict"]] = out.get(claim["verdict"], 0) + 1
        return out


# -- 1. cross-check --------------------------------------------------------


async def _cross_check_one(claim: dict, *, exa, ollama, budget: Budget, settings) -> dict:
    annotated = dict(claim)
    annotated.update(
        {
            "verdict": "kept",
            "reason": "",
            "support": "unknown",
            "corroboration": "absent",
            "source_quality": "unknown",
            "cross_check_sources": [],
        }
    )

    # Short-circuit. A claim whose cited URL the worker never retrieved is
    # rejected whatever Exa says, so paying for the cross-check would buy
    # nothing. On a typical run this is the single largest credit saving
    # available, and it costs no accuracy: the verdict is already determined.
    if not claim.get("source_url") or not claim.get("url_verified"):
        _apply_verdict(annotated)
        annotated["cross_check_skipped"] = "unverifiable citation — rejected without cross-check"
        return annotated

    own_domain = domain_of(claim.get("source_url", ""))

    independent: list[dict] = []
    try:
        independent = await asyncio.wait_for(
            exa.search(claim["text"], exclude_domains=[own_domain] if own_domain else None),
            timeout=settings.call_timeout_s,
        )
    except asyncio.TimeoutError:
        budget.record_failure("critic", "exa", f"cross-check timed out after {settings.call_timeout_s}s", claim_id=claim["id"])
    except Exception as exc:
        budget.record_failure("critic", "exa", f"cross-check failed: {exc}", claim_id=claim["id"])

    annotated["cross_check_sources"] = [s["url"] for s in independent]

    # Programmatic priors the model does not get to override downward.
    if own_domain in LOW_SIGNAL_DOMAINS:
        annotated["source_quality"] = "low"
    has_attribution = any(s.get("author") for s in independent if domain_of(s["url"]) == own_domain)

    prompt = _cross_check_prompt(claim, independent)
    try:
        message = await ollama.chat(
            [{"role": "system", "content": CROSS_CHECK_SYSTEM}, {"role": "user", "content": prompt}],
            timeout=settings.call_timeout_s,
            json_mode=True,
            retries=0,
        )
        parsed = extract_json(message.get("content") or "") or {}
    except OllamaError as exc:
        budget.record_failure("critic", "ollama", str(exc), claim_id=claim["id"])
        parsed = {}

    if isinstance(parsed, dict):
        annotated["support"] = _enum(parsed.get("support"), ("supported", "unsupported", "unknown"), "unknown")
        annotated["corroboration"] = _enum(
            parsed.get("corroboration"), ("corroborated", "contradicted", "absent"), "absent"
        )
        quality = _enum(parsed.get("source_quality"), ("ok", "low", "unknown"), "unknown")
        if annotated["source_quality"] != "low":
            annotated["source_quality"] = quality
        annotated["reason"] = truncate(parsed.get("reason") or "", 240)

    if has_attribution and annotated["source_quality"] == "unknown":
        annotated["source_quality"] = "ok"

    _apply_verdict(annotated)
    return annotated


def _cross_check_prompt(claim: dict, independent: list[dict]) -> str:
    lines = [
        f"CLAIM: {claim['text']}",
        f"CITED SOURCE: {claim.get('source_url') or '(none given)'}",
        f"WORKER CONFIDENCE: {claim.get('confidence', 'medium')} — {claim.get('confidence_note') or 'no note'}",
        f"CITED URL WAS ACTUALLY RETRIEVED BY THE WORKER: {'yes' if claim.get('url_verified') else 'NO'}",
        "",
        "INDEPENDENT SOURCES:",
    ]
    if not independent:
        lines.append("(none returned — treat corroboration as absent, do not guess)")
    for idx, src in enumerate(independent, start=1):
        author = src.get("author") or "no named author"
        lines.append(f"[{idx}] {src['title']} — {src['url']} ({author})")
        if src.get("text"):
            lines.append(f"    {truncate(src['text'], 600)}")
    return "\n".join(lines)


def _apply_verdict(annotated: dict) -> None:
    """Deterministic verdict from the model's three judgements plus the
    programmatic URL check. Kept out of the model's hands so the policy is
    auditable and identical across runs."""
    reason = annotated.get("reason") or ""

    if not annotated.get("source_url"):
        annotated["verdict"] = "rejected"
        annotated["reason"] = "no source URL attached to the claim"
        return
    if not annotated.get("url_verified"):
        annotated["verdict"] = "rejected"
        annotated["reason"] = "cited URL was never returned by the worker's own searches"
        return
    if annotated["support"] == "unsupported":
        annotated["verdict"] = "rejected"
        annotated["reason"] = reason or "cited source does not support the claim"
        return
    if annotated["corroboration"] == "contradicted":
        annotated["verdict"] = "flagged"
        annotated["reason"] = reason or "independent sources contradict this claim"
        return
    if annotated["source_quality"] == "low":
        annotated["verdict"] = "flagged"
        annotated["reason"] = reason or "source has no identifiable author or publication"
        return
    if annotated["support"] == "unknown" and annotated["corroboration"] == "absent":
        annotated["verdict"] = "flagged"
        annotated["reason"] = reason or "no independent corroboration and source text unavailable"
        return
    annotated["verdict"] = "kept"
    annotated["reason"] = ""


def _enum(value, allowed: tuple[str, ...], default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return default


# -- 2. contradictions -----------------------------------------------------


async def detect_contradictions(claims: list[dict], *, ollama, budget: Budget, settings) -> list[dict]:
    """Only claims from *different* sub-questions are compared — a worker
    contradicting itself inside one sub-question is a different bug. Embedding
    similarity narrows the candidate pairs so we make O(pairs-about-the-same-
    thing) model calls instead of O(n²)."""
    if len(claims) < 2:
        return []

    # Embeddings are the primary gate but not a hard dependency: if Ollama's
    # embed endpoint fails, contradiction checking degrades to the lexical gate
    # rather than silently switching itself off.
    vectors: list[list[float]] = []
    try:
        vectors = await ollama.embed([c["text"] for c in claims])
        if len(vectors) != len(claims):
            budget.record_failure("critic", "embeddings", "vector count mismatch")
            vectors = []
    except OllamaError as exc:
        budget.record_failure("critic", "embeddings", str(exc))

    scored: dict[tuple[int, int], float] = {}
    if vectors:
        for i, j, score in similar_pairs(vectors, settings.contradiction_sim_threshold):
            scored[(i, j)] = score
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            overlap = lexical_overlap(claims[i]["text"], claims[j]["text"])
            if overlap >= LEXICAL_OVERLAP_THRESHOLD:
                scored[(i, j)] = max(scored.get((i, j), 0.0), round(overlap, 4))

    candidates = sorted(
        (
            (i, j, score)
            for (i, j), score in scored.items()
            # Different workers only: a worker disagreeing with itself inside one
            # sub-question is a different defect.
            if claims[i]["sub_question"] != claims[j]["sub_question"]
            # Two workers quoting the same source produce the same claim twice.
            # That is duplication, not disagreement, and checking it wastes calls.
            and score < 0.995
            and claims[i]["text"].strip().lower() != claims[j]["text"].strip().lower()
        ),
        key=lambda t: t[2],
        reverse=True,
    )[: settings.max_contradiction_checks]  # keeps this from dominating the budget

    found: list[dict] = []
    semaphore = asyncio.Semaphore(settings.search_workers)

    async def check(i: int, j: int, score: float):
        prompt = (
            f"CLAIM A (from: {claims[i]['sub_question']})\n{claims[i]['text']}\n"
            f"source: {claims[i]['source_url']}\n\n"
            f"CLAIM B (from: {claims[j]['sub_question']})\n{claims[j]['text']}\n"
            f"source: {claims[j]['source_url']}"
        )
        async with semaphore:
            try:
                message = await ollama.chat(
                    [
                        {"role": "system", "content": CONTRADICTION_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    timeout=settings.call_timeout_s,
                    json_mode=True,
                    retries=0,
                )
            except OllamaError as exc:
                budget.record_failure("critic", "ollama-contradiction", str(exc))
                return
        parsed = extract_json(message.get("content") or "")
        if isinstance(parsed, dict) and parsed.get("contradiction") is True:
            found.append(
                {
                    "claim_ids": [claims[i]["id"], claims[j]["id"]],
                    "claim_texts": [claims[i]["text"], claims[j]["text"]],
                    "sources": [claims[i]["source_url"], claims[j]["source_url"]],
                    "similarity": score,
                    "explanation": truncate(parsed.get("explanation") or "claims conflict", 240),
                }
            )

    await asyncio.gather(*(check(i, j, s) for i, j, s in candidates))
    return found


# -- 3. review = cross-check + contradictions ------------------------------


async def review(claims: list[dict], *, exa, ollama, budget: Budget, settings) -> CriticResult:
    if not claims:
        return CriticResult()

    semaphore = asyncio.Semaphore(settings.search_workers)

    async def guarded(claim):
        async with semaphore:
            return await _cross_check_one(
                claim, exa=exa, ollama=ollama, budget=budget, settings=settings
            )

    annotated = list(await asyncio.gather(*(guarded(c) for c in claims)))
    contradictions = await detect_contradictions(
        annotated, ollama=ollama, budget=budget, settings=settings
    )

    by_id = {c["id"]: c for c in annotated}
    for conflict in contradictions:
        for claim_id in conflict["claim_ids"]:
            claim = by_id.get(claim_id)
            if claim and claim["verdict"] != "rejected":
                claim["verdict"] = "flagged"
                claim["reason"] = f"contradicts another worker's claim: {conflict['explanation']}"
                claim["corroboration"] = "contradicted"

    return CriticResult(annotated=annotated, contradictions=contradictions)


# -- 4. synthesis (the critic's own final call) ----------------------------


async def synthesize(query: str, kept: list[dict], *, ollama, budget: Budget, settings) -> dict:
    """Returns {'report': str, 'sources': [{index, url, title}]}.

    Citation numbers are assigned here, deterministically, before the model
    writes anything — the model is told the numbers rather than inventing them,
    and any number it emits outside the valid range is stripped afterwards.
    """
    if not kept:
        return {
            "report": (
                "No claims survived fact-checking, so there is no report to write. "
                "Check the run log for the critic's rejection reasons."
            ),
            "sources": [],
        }

    sources: list[dict] = []
    index_by_url: dict[str, int] = {}
    for claim in kept:
        url = claim.get("source_url", "")
        if url and url not in index_by_url:
            index_by_url[url] = len(sources) + 1
            sources.append({"index": len(sources) + 1, "url": url, "title": domain_of(url)})

    numbered = "\n".join(
        f"[{index_by_url.get(c.get('source_url',''), 0)}] {c['text']}  (source: {c.get('source_url','')})"
        for c in kept
    )
    prompt = (
        f"RESEARCH QUESTION: {query}\n\n"
        f"VERIFIED CLAIMS (cite by the bracketed number):\n{numbered}\n\n"
        "Write the report."
    )

    try:
        message = await ollama.chat(
            [{"role": "system", "content": SYNTH_SYSTEM}, {"role": "user", "content": prompt}],
            timeout=settings.call_timeout_s * 2,  # longest single generation in the run
            temperature=0.3,
            retries=1,
        )
        report = (message.get("content") or "").strip()
    except OllamaError as exc:
        budget.record_failure("synthesis", "ollama", str(exc))
        report = _fallback_report(query, kept, index_by_url)

    if len(report) < 80:
        report = _fallback_report(query, kept, index_by_url)

    report = _strip_invalid_citations(report, len(sources))
    return {"report": report, "sources": sources}


def _strip_invalid_citations(report: str, max_index: int) -> str:
    def repl(match: re.Match) -> str:
        try:
            return match.group(0) if 1 <= int(match.group(1)) <= max_index else ""
        except ValueError:
            return ""

    return re.sub(r"\[(\d{1,3})\]", repl, report)


def _fallback_report(query: str, kept: list[dict], index_by_url: dict[str, int]) -> str:
    """Deterministic report used when the model call fails. Keeps the run
    usable and keeps citations correct rather than emitting an empty string."""
    lines = [f"## {query}", "", "Synthesis was unavailable, so the verified claims are listed as found.", ""]
    by_sub: dict[str, list[dict]] = {}
    for claim in kept:
        by_sub.setdefault(claim.get("sub_question", "General"), []).append(claim)
    for sub, group in by_sub.items():
        lines.append(f"### {sub}")
        for claim in group:
            idx = index_by_url.get(claim.get("source_url", ""), 0)
            lines.append(f"- {claim['text']} [{idx}]" if idx else f"- {claim['text']}")
        lines.append("")
    return "\n".join(lines).strip()
