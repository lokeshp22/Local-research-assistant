#!/usr/bin/env python3
"""Offline self-test. No GPU, no network, no API keys.

Verifies the parts that are expensive to debug against a live stack:
defensive parsing, the planner's dedup/merge, the 2-worker queue, the critic's
verdict policy, citation numbering, the JSONL schema the dashboard reads, and
every Flask route. Uses the scripted clients in tools/fakes.py, so it proves
wiring and schemas — not answer quality.

    python selftest.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import load_settings  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from tools.budget import Budget  # noqa: E402
from tools.fakes import FakeExa, FakeOllama, FakeTavily, FixtureExaClient, FixtureTavilyClient  # noqa: E402
from tools.jsonl_log import read_runs  # noqa: E402
from tools.parsing import coerce_tool_args, extract_json, normalize_url, strip_think  # noqa: E402

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> bool:
    (PASS if condition else FAIL).append(name)
    mark = "  ok  " if condition else " FAIL "
    print(f"[{mark}] {name}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def test_parsing() -> None:
    print("\n— defensive parsing —")
    check("tool args as dict", coerce_tool_args({"q": 1}) == {"q": 1})
    check("tool args as JSON string", coerce_tool_args('{"query": "x"}') == {"query": "x"})
    check("tool args double-encoded", coerce_tool_args('"{\\"query\\": \\"x\\"}"') == {"query": "x"})
    check("tool args in code fence", coerce_tool_args('```json\n{"a": 2}\n```') == {"a": 2})
    check("tool args garbage -> {}", coerce_tool_args("not json at all") == {})
    check("tool args None -> {}", coerce_tool_args(None) == {})
    check("think block stripped", strip_think("<think>hmm</think>answer") == "answer")
    check("unterminated think stripped", strip_think("answer<think>hmm") == "answer")
    check("json after prose", extract_json('sure thing:\n{"claims": []}') == {"claims": []})
    check("url normalised", normalize_url("HTTP://WWW.Example.com/a/") == "http://example.com/a")
    check("bad url -> empty", normalize_url("::::") == "")


async def test_pipeline(settings) -> dict:
    print("\n— pipeline (scripted model) —")
    budget = Budget()
    ollama = FakeOllama(settings, budget, sub_questions=5)
    orchestrator = Orchestrator(
        settings,
        ollama=ollama,
        tavily=FakeTavily(settings, budget),
        exa=FakeExa(settings, budget),
        budget=budget,
    )
    record = await orchestrator.run("Does tiered caching actually reduce tail latency?")

    planner = record["planner"]
    check("planner produced sub-questions", len(planner["sub_questions"]) >= 3)
    check("sub-question cap respected", len(planner["raw_sub_questions"]) <= settings.max_subquestions)
    check(
        "near-duplicate sub-questions merged",
        len(planner["merges"]) >= 1 and len(planner["sub_questions"]) < len(planner["raw_sub_questions"]),
        f"{len(planner['raw_sub_questions'])} -> {len(planner['sub_questions'])}",
    )
    check("worker pool size is 2", record["worker_pool_size"] == 2)
    check("every sub-question got a result", len(record["sub_questions"]) == len(planner["sub_questions"]))
    check("claims were produced", len(record["claims"]) > 0)
    check("tool turns capped", all(s["turns"] <= settings.max_turns for s in record["sub_questions"]))

    verdicts = {c["verdict"] for c in record["claims"]}
    check("verdicts are from the allowed set", verdicts <= {"kept", "flagged", "rejected"})
    check("some claims kept", record["verdict_counts"]["kept"] > 0)
    check(
        "unverifiable citation rejected",
        any(c["verdict"] == "rejected" and not c["url_verified"] for c in record["claims"]),
    )
    check(
        "every non-kept claim has a reason",
        all(c["reason"] for c in record["claims"] if c["verdict"] != "kept"),
    )
    check("report is non-empty", len(record["final_report"]) > 80)
    check("sources list is non-empty", len(record["sources"]) > 0)

    import re

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", record["final_report"])}
    valid = {s["index"] for s in record["sources"]}
    check("every citation resolves to a source", cited <= valid, f"dangling: {sorted(cited - valid)}")
    check("citations actually present in report", len(cited) > 0)
    check(
        "kept claims all carry a verified URL",
        all(c["url_verified"] for c in record["claims"] if c["verdict"] == "kept"),
    )

    budget_d = record["budget"]
    check("tavily credits tracked", budget_d["tavily"]["credits_estimated"] > 0)
    check("exa credits tracked", budget_d["exa"]["credits_estimated"] > 0)
    check("ollama calls tracked", budget_d["ollama"]["chat_calls"] > 0)
    check("wall clock tracked", budget_d["wall_clock_s"] >= 0)
    check("stage timings present", set(record["timing_s"]) >= {"planner", "search", "critic", "synthesis", "total"})
    return record


async def test_graceful_degradation(settings) -> None:
    print("\n— graceful degradation —")
    budget = Budget()

    class BrokenTavily(FakeTavily):
        async def search(self, query, max_results=None):
            raise RuntimeError("429 rate limited")

        async def extract(self, urls):
            raise RuntimeError("upstream 503")

    class BrokenExa(FakeExa):
        async def search(self, query, num_results=None, exclude_domains=None):
            raise RuntimeError("exa down")

    orchestrator = Orchestrator(
        settings,
        ollama=FakeOllama(settings, budget),
        tavily=BrokenTavily(settings, budget),
        exa=BrokenExa(settings, budget),
        budget=budget,
    )
    record = await orchestrator.run("What happens when both providers are down?", log=False)
    check("run completed instead of crashing", isinstance(record, dict))
    check("failures were recorded", len(record["failures"]) > 0)
    check("status reflects the damage", record["status"] in {"degraded", "failed"})
    check("report field still present", "final_report" in record)


async def test_contradiction_fixtures(settings) -> None:
    print("\n— seeded contradiction tests —")
    cases = json.loads((settings.eval_dir / "contradiction_test_cases.json").read_text())["cases"]
    passed = 0
    for case in cases:
        budget = Budget()
        orchestrator = Orchestrator(
            settings,
            ollama=FakeOllama(settings, budget, sub_questions=3),
            tavily=FixtureTavilyClient(case["sources"], budget=budget),
            exa=FixtureExaClient(case["sources"], budget=budget),
            budget=budget,
        )
        record = await orchestrator.run(case["query"], run_id=f"ct_{case['id']}", log=False)
        hit = len(record.get("contradictions", [])) > 0
        passed += bool(hit)
        check(f"{case['id']} contradiction flagged", hit, case["conflict_summary"])
    check("all seeded contradictions caught", passed == len(cases), f"{passed}/{len(cases)}")


def test_log_and_routes(settings, record: dict) -> None:
    print("\n— run log and dashboard routes —")
    runs = read_runs(settings.run_log_path)
    check("run appended to JSONL", any(r["run_id"] == record["run_id"] for r in runs))
    required = {
        "schema_version", "run_id", "timestamp", "query", "status", "planner",
        "sub_questions", "claims", "contradictions", "final_report", "sources",
        "timing_s", "budget", "failures", "metrics", "verdict_counts",
    }
    check("log schema complete", required <= set(record), f"missing: {sorted(required - set(record))}")
    check("record round-trips as JSON", json.loads(json.dumps(record))["run_id"] == record["run_id"])

    sys.modules.pop("dashboard.app", None)
    from dashboard import app as dash

    dash.SETTINGS = settings
    dash._cache.clear()
    client = dash.app.test_client()

    resp = client.get("/")
    check("GET / serves the dashboard", resp.status_code == 200 and b"Research pipeline" in resp.data)

    resp = client.get("/api/runs")
    body = resp.get_json()
    check("GET /api/runs", resp.status_code == 200 and body["totals"]["runs"] >= 1)
    check("run summary has verdict counts", "kept" in body["runs"][0])

    resp = client.get(f"/api/run/{record['run_id']}")
    check("GET /api/run/<id>", resp.status_code == 200 and resp.get_json()["run_id"] == record["run_id"])

    resp = client.get("/api/run/does-not-exist")
    check("unknown run returns 404", resp.status_code == 404)

    resp = client.get("/api/eval-results")
    check("GET /api/eval-results answers", resp.status_code in (200, 404))

    check("GET /api/health", client.get("/api/health").status_code == 200)

    before = settings.run_log_path.read_bytes()
    client.get("/api/runs"); client.get(f"/api/run/{record['run_id']}"); client.get("/api/eval-results")
    check("dashboard never wrote to the log", settings.run_log_path.read_bytes() == before)


async def main() -> int:
    settings = load_settings()
    with tempfile.TemporaryDirectory() as tmp:
        settings = type(settings)(
            **{**settings.__dict__, "run_log_path": Path(tmp) / "runs.jsonl", "num_parallel": 2}
        )
        test_parsing()
        record = await test_pipeline(settings)
        await test_graceful_degradation(settings)
        await test_contradiction_fixtures(settings)
        test_log_and_routes(settings, record)

    print("\n" + "=" * 62)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
