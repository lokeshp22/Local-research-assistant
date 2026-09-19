#!/usr/bin/env python3
"""Eval harness.

Runs every held-out question through two systems and scores both the same way:

  baseline  — one Tavily search bolted onto one model call. No planner, no
              critic, no cross-checking. This is the thing the multi-agent
              pipeline has to beat to justify its credits.
  pipeline  — the full planner -> 2 workers -> critic -> synthesis run.

Scoring is deliberately split:
  * citation coverage and source diversity are computed **programmatically**
    from the run record — a citation only counts if its number resolves to a
    source and that URL was actually retrieved by a tool during the run;
  * depth/completeness is the one axis left to the LLM judge, which sees the
    two reports in randomised order so it cannot learn a position bias.

Contradiction tests are separate and are not judged by a model at all: fixture
clients inject conflicting sources, and the case passes only if the critic
flags a contradiction between claims drawn from the conflicting pair.

    python eval/run_eval.py                 # full run against real services
    python eval/run_eval.py --limit 3       # smoke test
    python eval/run_eval.py --fake          # offline plumbing check, no GPU
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import ConfigError, load_settings  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from tools.budget import Budget  # noqa: E402
from tools.exa_client import ExaClient  # noqa: E402
from tools.fakes import FakeExa, FakeOllama, FakeTavily, FixtureExaClient, FixtureTavilyClient  # noqa: E402
from tools.metrics import citation_metrics  # noqa: E402
from tools.ollama_client import OllamaClient, OllamaError  # noqa: E402
from tools.parsing import domain_of, extract_json, truncate  # noqa: E402
from tools.tavily_client import TavilyClient  # noqa: E402

BASELINE_SYSTEM = """You are a baseline research assistant. You get one batch of web search
results and one chance to answer.

Write a report answering the question using only those results. Cite inline with
[1], [2] matching the numbered results you were given. Every substantive sentence
carries a citation. Do not write a source list; it is appended for you."""

JUDGE_SYSTEM = """You are a JUDGE scoring research reports against a fixed rubric.

Score depth and completeness on 1-5:
  1 — restates the question, almost no substance
  2 — one thin angle, obvious gaps
  3 — covers the main angle competently, misses counterpoints
  4 — several angles including at least one competing view, few gaps
  5 — thorough: mechanism, current evidence, competing views, and limits

Judge substance only. Ignore length, formatting, and confident tone. Do not
reward citations here — citation quality is measured separately and
programmatically.

Reply with JSON only: {"depth": 1-5, "reason": "one short line"}"""


# ----------------------------------------------------------------- baseline


async def run_baseline(question: str, *, ollama, tavily, settings, budget: Budget) -> dict:
    started = time.monotonic()
    failures: list[dict] = []
    results: list[dict] = []
    try:
        results = await asyncio.wait_for(
            tavily.search(question, max_results=settings.tavily_max_results),
            timeout=settings.call_timeout_s,
        )
    except Exception as exc:
        failures.append(budget.record_failure("baseline", "tavily", str(exc)))

    sources = [
        {"index": i + 1, "url": r["url"], "title": r.get("title", "")}
        for i, r in enumerate(results)
    ]
    numbered = "\n".join(
        f"[{s['index']}] {s['title']} — {s['url']}\n    {truncate(results[i].get('snippet',''), 700)}"
        for i, s in enumerate(sources)
    )
    prompt = f"QUESTION: {question}\n\nSEARCH RESULTS:\n{numbered or '(search failed)'}\n\nWrite the report."

    report = ""
    try:
        message = await ollama.chat(
            [{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": prompt}],
            timeout=settings.call_timeout_s * 2,
            temperature=0.3,
            retries=0,
        )
        report = (message.get("content") or "").strip()
    except OllamaError as exc:
        failures.append(budget.record_failure("baseline", "ollama", str(exc)))

    return {
        "approach": "baseline",
        "question": question,
        "final_report": report,
        "sources": sources,
        "retrieved_urls": [r["url"] for r in results],
        "latency_s": round(time.monotonic() - started, 3),
        "failures": failures,
    }


# -------------------------------------------------------------- llm judge


async def judge(question: str, report_a: dict, report_b: dict, *, ollama, settings, budget: Budget) -> dict:
    """Randomised A/B presentation; returns depth per approach."""
    pair = [report_a, report_b]
    random.shuffle(pair)
    labels = {"A": pair[0]["approach"], "B": pair[1]["approach"]}
    scores: dict[str, int] = {}
    reasons: dict[str, str] = {}

    for label, item in zip(("A", "B"), pair):
        prompt = (
            f"QUESTION: {question}\n\nREPORT {label}:\n{truncate(item['final_report'], 6000) or '(empty report)'}"
        )
        depth, reason = 1, "empty or unscorable report"
        if item["final_report"].strip():
            try:
                message = await ollama.chat(
                    [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
                    timeout=settings.call_timeout_s * 2,
                    json_mode=True,
                    retries=0,
                )
                parsed = extract_json(message.get("content") or "") or {}
                raw = parsed.get("depth")
                depth = int(raw) if isinstance(raw, (int, float, str)) and str(raw).strip().isdigit() else 1
                depth = max(1, min(5, depth))
                reason = truncate(parsed.get("reason") or "", 200)
            except (OllamaError, ValueError, TypeError) as exc:
                budget.record_failure("judge", "ollama", str(exc))
        scores[labels[label]] = depth
        reasons[labels[label]] = reason

    return {"depth": scores, "reason": reasons, "presentation_order": labels}


# ------------------------------------------------------ contradiction tests


async def run_contradiction_tests(cases: list[dict], *, ollama, settings) -> list[dict]:
    out: list[dict] = []
    for case in cases:
        budget = Budget()
        tavily = FixtureTavilyClient(case["sources"], budget=budget)
        exa = FixtureExaClient(case["sources"], budget=budget)
        if hasattr(ollama, "budget"):
            ollama.budget = budget
        orchestrator = Orchestrator(settings, ollama=ollama, tavily=tavily, exa=exa, budget=budget)

        try:
            record = await orchestrator.run(case["query"], run_id=f"ct_{case['id']}", log=False)
        except Exception as exc:
            out.append(
                {"id": case["id"], "passed": False, "detail": f"run crashed: {type(exc).__name__}: {exc}"}
            )
            continue

        fixture_domains = {domain_of(s["url"]) for s in case["sources"]}
        # Pass only if a contradiction was raised between claims that came from
        # two DIFFERENT injected sources — flagging a claim against itself, or
        # flagging for source quality, does not count.
        hit = None
        for conflict in record.get("contradictions", []):
            domains = {domain_of(u) for u in conflict.get("sources", []) if u}
            if len(domains & fixture_domains) >= 2:
                hit = conflict
                break

        flagged_ids = {c["id"] for c in record["claims"] if c["verdict"] == "flagged"}
        out.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expect": case.get("expect", "flagged"),
                "passed": hit is not None,
                "detail": hit["explanation"] if hit else "critic did not flag a cross-source contradiction",
                "contradictions_found": len(record.get("contradictions", [])),
                "claims_total": len(record["claims"]),
                "claims_flagged": len(flagged_ids),
                "conflict_summary": case.get("conflict_summary", ""),
            }
        )
    return out


def _bind(clients, budget: Budget) -> None:
    """Point long-lived clients at a specific run's budget."""
    for client in clients:
        if hasattr(client, "budget"):
            client.budget = budget


# ------------------------------------------------------------------ driver


async def main_async(args) -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    questions_path = Path(args.questions) if args.questions else settings.eval_dir / "eval_questions.json"
    cases_path = Path(args.cases) if args.cases else settings.eval_dir / "contradiction_test_cases.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    if args.limit:
        questions = questions[: args.limit]
        cases = cases[: args.limit]

    random.seed(args.seed)
    budget = Budget()

    if args.fake:
        print("[eval] --fake: scripted model and search clients. Plumbing check only; scores are not meaningful.", file=sys.stderr)
        ollama = FakeOllama(settings, budget)
        make_tavily = lambda: FakeTavily(settings, budget)  # noqa: E731
        make_exa = lambda: FakeExa(settings, budget)  # noqa: E731
    else:
        try:
            settings.require_search_keys()
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        settings.warn_if_oversized()
        ollama = OllamaClient(settings, budget)
        try:
            await ollama.preflight()
        except OllamaError as exc:
            print(f"ollama error: {exc}", file=sys.stderr)
            await ollama.aclose()
            return 3
        make_tavily = lambda: TavilyClient(settings, budget)  # noqa: E731
        make_exa = lambda: ExaClient(settings, budget)  # noqa: E731

    tavily, exa = make_tavily(), make_exa()
    rows: list[dict] = []
    started = time.monotonic()

    try:
        for i, item in enumerate(questions, start=1):
            question = item["question"]
            print(f"[eval] {i}/{len(questions)} {item['id']}: {truncate(question, 70)}", file=sys.stderr)

            run_budget = Budget()
            # The clients are long-lived but the accounting is per-question, so
            # point them at this run's budget before the run. Without this the
            # log records zero credits for every eval run and the "is the
            # pipeline worth its cost" comparison is unanswerable.
            _bind((ollama, tavily, exa), run_budget)
            orchestrator = Orchestrator(
                settings,
                ollama=ollama,
                tavily=tavily,
                exa=exa,
                budget=run_budget,
                mode="offline" if args.fake else "live",
            )
            try:
                record = await orchestrator.run(question, log=not args.no_log)
            except Exception as exc:
                print(f"  pipeline crashed: {exc}", file=sys.stderr)
                record = {
                    "run_id": f"failed_{item['id']}", "status": "failed", "final_report": "",
                    "sources": [], "claims": [], "sub_questions": [], "contradictions": [],
                    "verdict_counts": {}, "timing_s": {"total": 0.0},
                    "budget": run_budget.as_dict(), "failures": run_budget.failures,
                    "metrics": {},
                }

            pipeline_out = {
                "approach": "multi_agent",
                "question": question,
                "final_report": record["final_report"],
                "sources": record["sources"],
                "retrieved_urls": sorted(
                    {url for sq in record.get("sub_questions", []) for url in sq.get("sources_seen", [])}
                ),
            }

            baseline_budget = Budget()
            _bind((ollama, tavily, exa), baseline_budget)
            baseline_out = await run_baseline(
                question, ollama=ollama, tavily=tavily, settings=settings, budget=baseline_budget
            )

            _bind((ollama, tavily, exa), budget)  # judge calls bill to the harness

            pipeline_metrics = citation_metrics(
                pipeline_out["final_report"], pipeline_out["sources"], pipeline_out["retrieved_urls"]
            )
            baseline_metrics = citation_metrics(
                baseline_out["final_report"], baseline_out["sources"], baseline_out["retrieved_urls"]
            )
            verdict = await judge(
                question, pipeline_out, baseline_out, ollama=ollama, settings=settings, budget=budget
            )

            rows.append(
                {
                    "id": item["id"],
                    "domain": item.get("domain", ""),
                    "complexity": item.get("complexity", ""),
                    "question": question,
                    "run_id": record.get("run_id", ""),
                    "multi_agent": {
                        **pipeline_metrics,
                        "depth": verdict["depth"].get("multi_agent", 0),
                        "depth_reason": verdict["reason"].get("multi_agent", ""),
                        "latency_s": record.get("timing_s", {}).get("total", 0.0),
                        "status": record.get("status", "unknown"),
                        "claims_kept": record.get("verdict_counts", {}).get("kept", 0),
                        "claims_flagged": record.get("verdict_counts", {}).get("flagged", 0),
                        "claims_rejected": record.get("verdict_counts", {}).get("rejected", 0),
                        "tavily_credits": record.get("budget", {}).get("tavily", {}).get("credits_estimated", 0),
                        "exa_credits": record.get("budget", {}).get("exa", {}).get("credits_estimated", 0),
                        "ollama_calls": record.get("budget", {}).get("ollama", {}).get("chat_calls", 0),
                        "failures": len(record.get("failures", [])),
                    },
                    "baseline": {
                        **baseline_metrics,
                        "depth": verdict["depth"].get("baseline", 0),
                        "depth_reason": verdict["reason"].get("baseline", ""),
                        "latency_s": baseline_out["latency_s"],
                        "status": "ok" if baseline_out["final_report"] else "failed",
                        "failures": len(baseline_out["failures"]),
                        "tavily_credits": baseline_budget.as_dict()["tavily"]["credits_estimated"],
                        "exa_credits": baseline_budget.as_dict()["exa"]["credits_estimated"],
                        "ollama_calls": baseline_budget.as_dict()["ollama"]["chat_calls"],
                    },
                }
            )

        print(f"[eval] contradiction tests: {len(cases)} case(s)", file=sys.stderr)
        contradiction_results = await run_contradiction_tests(cases, ollama=ollama, settings=settings)
    finally:
        for client in (ollama, tavily, exa):
            close = getattr(client, "aclose", None)
            if close:
                await close()

    passed = sum(1 for r in contradiction_results if r["passed"])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "fake" if args.fake else "live",
        "model": settings.ollama_model,
        "question_count": len(rows),
        "wall_clock_s": round(time.monotonic() - started, 2),
        "aggregate": _aggregate(rows),
        "contradiction_tests": {
            "passed": passed,
            "total": len(contradiction_results),
            "summary": f"{passed}/{len(contradiction_results)} passed",
            "cases": contradiction_results,
        },
        "results": rows,
    }

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out) if args.out else settings.results_dir / "eval_results.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_csv = out_json.with_suffix(".csv")
    _write_csv(out_csv, rows)

    print(f"\n[eval] wrote {out_json}", file=sys.stderr)
    print(f"[eval] wrote {out_csv}", file=sys.stderr)
    _print_summary(payload)
    return 0


def _aggregate(rows: list[dict]) -> dict:
    def mean(approach: str, key: str) -> float:
        values = [r[approach].get(key, 0) or 0 for r in rows]
        return round(sum(values) / len(values), 4) if values else 0.0

    out = {}
    for approach in ("multi_agent", "baseline"):
        out[approach] = {
            "depth_mean": mean(approach, "depth"),
            "citation_coverage_mean": mean(approach, "citation_coverage"),
            "sentence_citation_rate_mean": mean(approach, "sentence_citation_rate"),
            "source_diversity_mean": mean(approach, "source_diversity"),
            "latency_s_mean": mean(approach, "latency_s"),
            "tavily_credits_mean": mean(approach, "tavily_credits"),
            "exa_credits_mean": mean(approach, "exa_credits"),
            "ollama_calls_mean": mean(approach, "ollama_calls"),
            "empty_reports": sum(1 for r in rows if not r[approach].get("report_chars")),
        }
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "id", "domain", "complexity", "question",
        "ma_depth", "ma_citation_coverage", "ma_sentence_citation_rate", "ma_source_diversity",
        "ma_latency_s", "ma_claims_kept", "ma_claims_flagged", "ma_claims_rejected",
        "ma_tavily_credits", "ma_exa_credits", "ma_ollama_calls", "ma_status",
        "bl_depth", "bl_citation_coverage", "bl_sentence_citation_rate", "bl_source_diversity",
        "bl_latency_s", "bl_tavily_credits", "bl_exa_credits", "bl_ollama_calls", "bl_status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            ma, bl = row["multi_agent"], row["baseline"]
            writer.writerow(
                {
                    "id": row["id"], "domain": row["domain"], "complexity": row["complexity"],
                    "question": row["question"],
                    "ma_depth": ma["depth"], "ma_citation_coverage": ma["citation_coverage"],
                    "ma_sentence_citation_rate": ma["sentence_citation_rate"],
                    "ma_source_diversity": ma["source_diversity"], "ma_latency_s": ma["latency_s"],
                    "ma_claims_kept": ma["claims_kept"], "ma_claims_flagged": ma["claims_flagged"],
                    "ma_claims_rejected": ma["claims_rejected"],
                    "ma_tavily_credits": ma["tavily_credits"], "ma_exa_credits": ma["exa_credits"],
                    "ma_ollama_calls": ma["ollama_calls"], "ma_status": ma["status"],
                    "bl_depth": bl["depth"], "bl_citation_coverage": bl["citation_coverage"],
                    "bl_sentence_citation_rate": bl["sentence_citation_rate"],
                    "bl_source_diversity": bl["source_diversity"], "bl_latency_s": bl["latency_s"],
                    "bl_tavily_credits": bl.get("tavily_credits", 0),
                    "bl_exa_credits": bl.get("exa_credits", 0),
                    "bl_ollama_calls": bl.get("ollama_calls", 0),
                    "bl_status": bl["status"],
                }
            )


def _print_summary(payload: dict) -> None:
    agg = payload["aggregate"]
    print("\n" + "=" * 68)
    print(f"EVAL SUMMARY  ({payload['mode']} mode, {payload['question_count']} questions)")
    print("=" * 68)
    print(f"{'metric':<30}{'multi-agent':>18}{'baseline':>18}")
    for label, key in (
        ("depth (1-5)", "depth_mean"),
        ("citation coverage", "citation_coverage_mean"),
        ("cited-sentence rate", "sentence_citation_rate_mean"),
        ("source diversity", "source_diversity_mean"),
        ("latency (s)", "latency_s_mean"),
        ("Tavily credits/question", "tavily_credits_mean"),
        ("Exa credits/question", "exa_credits_mean"),
        ("Ollama calls/question", "ollama_calls_mean"),
        ("empty reports", "empty_reports"),
    ):
        print(f"{label:<30}{agg['multi_agent'][key]:>18}{agg['baseline'][key]:>18}")
    print("-" * 68)
    print(f"contradiction tests: {payload['contradiction_tests']['summary']}")
    for case in payload["contradiction_tests"]["cases"]:
        mark = "PASS" if case["passed"] else "FAIL"
        print(f"  [{mark}] {case['id']}: {case['detail']}")
    print("=" * 68)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_eval.py", description="Baseline vs multi-agent eval.")
    parser.add_argument("--questions", help="Path to eval_questions.json")
    parser.add_argument("--cases", help="Path to contradiction_test_cases.json")
    parser.add_argument("--out", help="Output JSON path (CSV written alongside)")
    parser.add_argument("--limit", type=int, help="Only run the first N questions and cases")
    parser.add_argument("--seed", type=int, default=7, help="Seed for judge presentation order")
    parser.add_argument("--fake", action="store_true", help="Offline plumbing run: scripted model, no network")
    parser.add_argument("--no-log", action="store_true", help="Do not append eval runs to the JSONL log")
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(build_parser().parse_args())))
