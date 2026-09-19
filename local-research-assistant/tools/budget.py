"""Per-query budget accounting.

Credit costs are an accounting *model*, not something the APIs report back on
every call. The constants below come from the providers' published pricing at
build time and are deliberately in one place so they can be corrected against
your real usage dashboard without touching any other file.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Tavily: 1 credit per basic search, 2 per advanced search.
TAVILY_SEARCH_CREDITS = {"basic": 1.0, "advanced": 2.0}
# Tavily extract: billed per 5 URLs, rounded up.
TAVILY_EXTRACT_CREDITS_PER_BATCH = {"basic": 1.0, "advanced": 2.0}
TAVILY_EXTRACT_BATCH_SIZE = 5

# Exa: 1 credit per search request; retrieving page text is billed per result.
EXA_SEARCH_CREDITS = 1.0
EXA_TEXT_CREDITS_PER_RESULT = 0.2


@dataclass
class Budget:
    """Thread/task-safe counters for one pipeline run."""

    started_at: float = field(default_factory=time.monotonic)

    tavily_searches: int = 0
    tavily_extract_calls: int = 0
    tavily_extract_urls: int = 0
    tavily_credits: float = 0.0

    exa_searches: int = 0
    exa_results: int = 0
    exa_credits: float = 0.0

    ollama_calls: int = 0
    ollama_embed_calls: int = 0
    ollama_total_ms: float = 0.0

    failures: list[dict[str, Any]] = field(default_factory=list)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- recorders ---------------------------------------------------------
    def record_tavily_search(self, depth: str = "basic") -> None:
        with self._lock:
            self.tavily_searches += 1
            self.tavily_credits += TAVILY_SEARCH_CREDITS.get(depth, 1.0)

    def record_tavily_extract(self, n_urls: int, depth: str = "basic") -> None:
        n_urls = max(0, int(n_urls))
        batches = math.ceil(n_urls / TAVILY_EXTRACT_BATCH_SIZE) if n_urls else 0
        with self._lock:
            self.tavily_extract_calls += 1
            self.tavily_extract_urls += n_urls
            self.tavily_credits += batches * TAVILY_EXTRACT_CREDITS_PER_BATCH.get(depth, 1.0)

    def record_exa_search(self, n_results: int = 0, with_text: bool = True) -> None:
        n_results = max(0, int(n_results))
        with self._lock:
            self.exa_searches += 1
            self.exa_results += n_results
            self.exa_credits += EXA_SEARCH_CREDITS
            if with_text:
                self.exa_credits += n_results * EXA_TEXT_CREDITS_PER_RESULT

    def record_ollama(self, elapsed_ms: float, embed: bool = False) -> None:
        with self._lock:
            if embed:
                self.ollama_embed_calls += 1
            else:
                self.ollama_calls += 1
            self.ollama_total_ms += float(elapsed_ms)

    def record_failure(self, stage: str, component: str, detail: str, **extra: Any) -> dict:
        entry = {
            "stage": stage,
            "component": component,
            "detail": str(detail)[:500],
            "at_s": round(self.elapsed_s, 3),
        }
        entry.update(extra)
        with self._lock:
            self.failures.append(entry)
        return entry

    # -- readers -----------------------------------------------------------
    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_clock_s": round(self.elapsed_s, 3),
            "tavily": {
                "searches": self.tavily_searches,
                "extract_calls": self.tavily_extract_calls,
                "extract_urls": self.tavily_extract_urls,
                "credits_estimated": round(self.tavily_credits, 3),
            },
            "exa": {
                "searches": self.exa_searches,
                "results": self.exa_results,
                "credits_estimated": round(self.exa_credits, 3),
            },
            "ollama": {
                "chat_calls": self.ollama_calls,
                "embed_calls": self.ollama_embed_calls,
                "total_model_ms": round(self.ollama_total_ms, 1),
            },
            "failure_count": len(self.failures),
        }

    def summary_line(self) -> str:
        d = self.as_dict()
        return (
            f"{d['wall_clock_s']}s wall · "
            f"Tavily {d['tavily']['credits_estimated']} cr "
            f"({d['tavily']['searches']} search, {d['tavily']['extract_calls']} extract) · "
            f"Exa {d['exa']['credits_estimated']} cr ({d['exa']['searches']} search) · "
            f"Ollama {d['ollama']['chat_calls']} chat + {d['ollama']['embed_calls']} embed · "
            f"{d['failure_count']} failure(s)"
        )
