"""Citation and diversity metrics.

Shared by the orchestrator and the eval harness on purpose: "citation coverage"
must mean exactly one thing, or the number on the dashboard and the number in
the eval CSV quietly disagree.

Definition: a citation counts as *verifiable* when its bracketed number
resolves to an entry in the report's source list AND that URL was actually
returned by a tool during the run. The second half is the part that matters —
without it, a model that invents a plausible URL scores the same as one that
read the page.
"""

from __future__ import annotations

import re

from tools.parsing import domain_of

_CITE_RE = re.compile(r"\[(\d{1,3})\]")


def citation_metrics(report: str, sources: list[dict], retrieved_urls) -> dict:
    report = report or ""
    used = [int(n) for n in _CITE_RE.findall(report)]
    by_index = {s.get("index"): s.get("url", "") for s in (sources or [])}
    retrieved = set(retrieved_urls or [])

    resolvable = [n for n in used if n in by_index]
    verifiable = [n for n in resolvable if by_index[n] in retrieved]
    cited_urls = {by_index[n] for n in verifiable}
    domains = {d for d in (domain_of(u) for u in cited_urls) if d}

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", report) if len(s.strip()) > 40]
    cited_sentences = [s for s in sentences if _CITE_RE.search(s)]

    return {
        "citations_used": len(used),
        "citations_resolvable": len(resolvable),
        "citations_verifiable": len(verifiable),
        "citation_coverage": round(len(verifiable) / len(used), 4) if used else 0.0,
        "sentence_citation_rate": round(len(cited_sentences) / len(sentences), 4) if sentences else 0.0,
        "source_diversity": len(domains),
        "distinct_sources_cited": len(cited_urls),
        "report_chars": len(report),
    }
