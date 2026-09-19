"""Orchestrator — the 2-worker pipeline.

Why a queue and not asyncio.gather over the sub-questions:

Ollama serves OLLAMA_NUM_PARALLEL requests at a time and queues the rest. It
has no continuous batching, so a third concurrent generation does not share a
forward pass with the first two — it waits, and meanwhile its KV cache still
wants VRAM we do not have. On 8 GB that is the difference between a run that
finishes and one that thrashes.

So the fan-out is bounded at the source: N sub-questions go onto a queue, and
exactly `settings.search_workers` (= OLLAMA_NUM_PARALLEL = 2) workers pull from
it. A worker that finishes early immediately takes the next sub-question, which
is why this is *pipelined* rather than batched — there is no barrier between
sub-questions, only the drain of the queue before the critic starts.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

from agents import critic_agent
from agents.planner import PlannerError, plan
from agents.search_agent import run_sub_question
from tools.budget import Budget
from tools.jsonl_log import SCHEMA_VERSION, append_run
from tools.metrics import citation_metrics
from tools.parsing import domain_of


class Orchestrator:
    def __init__(self, settings, *, ollama, tavily, exa, budget: Budget | None = None, mode: str = "live"):
        self.s = settings
        # "live" or "offline" — offline runs come from scripted clients and must
        # never be mistaken for a real measurement on the dashboard.
        self.mode = mode
        self.ollama = ollama
        self.tavily = tavily
        self.exa = exa
        self.budget = budget or Budget()

    async def run(self, query: str, *, run_id: str | None = None, log: bool = True) -> dict:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        started_wall = time.monotonic()
        timing: dict[str, float] = {}

        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query": query,
            "status": "ok",
            "mode": self.mode,
            "model": self.s.ollama_model,
            "embed_model": self.s.embed_model,
            "worker_pool_size": self.s.search_workers,
            "planner": {},
            "sub_questions": [],
            "claims": [],
            "contradictions": [],
            "verdict_counts": {"kept": 0, "flagged": 0, "rejected": 0},
            "final_report": "",
            "sources": [],
            "timing_s": {},
            "budget": {},
            "failures": [],
        }

        # -- 1. plan -------------------------------------------------------
        stage = time.monotonic()
        try:
            plan_result = await plan(query, self.ollama, self.budget, self.s)
        except PlannerError as exc:
            timing["planner"] = time.monotonic() - stage
            self.budget.record_failure("planner", "fatal", str(exc))
            record.update(
                status="failed",
                final_report="",
                timing_s=_round(timing | {"total": time.monotonic() - started_wall}),
                budget=self.budget.as_dict(),
                failures=self.budget.failures,
            )
            if log:
                append_run(self.s.run_log_path, record)
            return record
        timing["planner"] = time.monotonic() - stage
        record["planner"] = plan_result.as_dict()

        # -- 2. search, 2 workers over a queue -----------------------------
        stage = time.monotonic()
        sub_results = await self._drain_queue(plan_result.sub_questions)
        timing["search"] = time.monotonic() - stage

        record["sub_questions"] = [r.as_dict() for r in sub_results]
        claims = [c.as_dict() for r in sub_results for c in r.claims]

        # -- 3. critic -----------------------------------------------------
        stage = time.monotonic()
        critique = await critic_agent.review(
            claims, exa=self.exa, ollama=self.ollama, budget=self.budget, settings=self.s
        )
        timing["critic"] = time.monotonic() - stage
        record["claims"] = critique.annotated
        record["contradictions"] = critique.contradictions
        record["verdict_counts"] = critique.counts()

        # -- 4. synthesis (critic's final call) ----------------------------
        stage = time.monotonic()
        synthesis = await critic_agent.synthesize(
            query, critique.kept, ollama=self.ollama, budget=self.budget, settings=self.s
        )
        timing["synthesis"] = time.monotonic() - stage
        record["final_report"] = synthesis["report"]
        record["sources"] = synthesis["sources"]

        # -- 5. finish -----------------------------------------------------
        timing["total"] = time.monotonic() - started_wall
        record["timing_s"] = _round(timing)
        record["budget"] = self.budget.as_dict()
        record["failures"] = self.budget.failures
        record["metrics"] = self._metrics(record)
        record["status"] = self._status(record)

        if log:
            append_run(self.s.run_log_path, record)
        return record

    # -- the pipeline itself ----------------------------------------------
    async def _drain_queue(self, sub_questions: list[str]):
        queue: asyncio.Queue = asyncio.Queue()
        for sub in sub_questions:
            queue.put_nowait(sub)

        results: list = []
        results_lock = asyncio.Lock()

        async def worker(worker_id: int):
            while True:
                try:
                    sub = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    try:
                        result = await run_sub_question(
                            sub,
                            ollama=self.ollama,
                            tavily=self.tavily,
                            budget=self.budget,
                            settings=self.s,
                        )
                    except Exception as exc:  # a worker must never take the run down
                        from agents.search_agent import SubQuestionResult

                        detail = f"worker {worker_id} crashed: {type(exc).__name__}: {exc}"
                        result = SubQuestionResult(
                            sub_question=sub,
                            status="failed",
                            failures=[self.budget.record_failure("search", "worker", detail)],
                        )
                    async with results_lock:
                        results.append(result)
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker(i) for i in range(self.s.search_workers)))

        order = {sub: idx for idx, sub in enumerate(sub_questions)}
        results.sort(key=lambda r: order.get(r.sub_question, 999))
        return results

    # -- derived metrics the eval harness and dashboard both read ----------
    @staticmethod
    def _metrics(record: dict) -> dict:
        claims = record["claims"]
        kept = [c for c in claims if c["verdict"] == "kept"]
        retrieved = {url for sq in record["sub_questions"] for url in sq.get("sources_seen", [])}
        # Same function the eval harness uses, so "citation coverage" on the
        # dashboard and in the eval CSV are the same number.
        metrics = citation_metrics(record["final_report"], record["sources"], retrieved)
        metrics.update(
            {
                "claim_count": len(claims),
                "kept_count": len(kept),
                # Distinct from citation_coverage: this is the share of *claims*
                # whose cited URL the worker genuinely retrieved.
                "claim_url_verification": round(
                    sum(1 for c in claims if c.get("url_verified")) / len(claims), 4
                )
                if claims
                else 0.0,
                "domains": sorted({d for d in (domain_of(s["url"]) for s in record["sources"]) if d}),
                "retrieved_urls": sorted(retrieved),
            }
        )
        return metrics

    @staticmethod
    def _status(record: dict) -> str:
        if not record["final_report"]:
            return "failed"
        if not record["claims"]:
            return "failed"
        if record["failures"] or any(
            s["status"] != "ok" for s in record["sub_questions"]
        ):
            return "degraded"
        return "ok"


def _round(timing: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 3) for k, v in timing.items()}
