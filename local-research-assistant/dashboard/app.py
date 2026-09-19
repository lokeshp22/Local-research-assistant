#!/usr/bin/env python3
"""Read-only Flask dashboard.

This process never writes to logs/runs.jsonl or to eval/results/ — the
orchestrator and the eval harness are the only writers. Every file here is
opened in read mode and there is no route that accepts a write.

    python dashboard/app.py            # http://127.0.0.1:5000
    python dashboard/app.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template  # noqa: E402

from config.settings import load_settings  # noqa: E402
from tools.jsonl_log import read_runs  # noqa: E402
from tools.parsing import domain_of  # noqa: E402

app = Flask(__name__)
SETTINGS = load_settings()

# Small mtime-keyed cache: the log is re-read only when it actually changes.
_cache: dict[str, tuple[float, object]] = {}


def _cached(path: Path, loader):
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    key = str(path)
    hit = _cache.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    value = loader(path)
    _cache[key] = (stamp, value)
    return value


def _load_runs() -> list[dict]:
    return _cached(SETTINGS.run_log_path, lambda p: read_runs(p)) or []


def _load_eval() -> dict | None:
    path = SETTINGS.results_dir / "eval_results.json"
    return _cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))


def _summarize(record: dict) -> dict:
    counts = record.get("verdict_counts") or {}
    metrics = record.get("metrics") or {}
    budget = record.get("budget") or {}
    return {
        "run_id": record.get("run_id", ""),
        "timestamp": record.get("timestamp", ""),
        "query": record.get("query", ""),
        "status": record.get("status", "unknown"),
        "model": record.get("model", ""),
        "mode": record.get("mode", "live"),
        "sub_question_count": len(record.get("sub_questions") or []),
        "merge_count": len((record.get("planner") or {}).get("merges") or []),
        "kept": counts.get("kept", 0),
        "flagged": counts.get("flagged", 0),
        "rejected": counts.get("rejected", 0),
        "claim_count": metrics.get("claim_count", len(record.get("claims") or [])),
        "citation_coverage": metrics.get("citation_coverage", 0),
        "source_diversity": metrics.get("source_diversity", len({domain_of(s.get("url", "")) for s in record.get("sources") or []} - {""})),
        "contradictions": len(record.get("contradictions") or []),
        "latency_s": (record.get("timing_s") or {}).get("total", 0),
        "tavily_credits": (budget.get("tavily") or {}).get("credits_estimated", 0),
        "exa_credits": (budget.get("exa") or {}).get("credits_estimated", 0),
        "ollama_calls": (budget.get("ollama") or {}).get("chat_calls", 0),
        "failure_count": len(record.get("failures") or []),
        "report_chars": len(record.get("final_report") or ""),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/runs")
def api_runs():
    runs = [_summarize(r) for r in _load_runs()]
    runs.sort(key=lambda r: r["timestamp"], reverse=True)
    totals = {
        "runs": len(runs),
        "kept": sum(r["kept"] for r in runs),
        "flagged": sum(r["flagged"] for r in runs),
        "rejected": sum(r["rejected"] for r in runs),
        "tavily_credits": round(sum(r["tavily_credits"] for r in runs), 2),
        "exa_credits": round(sum(r["exa_credits"] for r in runs), 2),
        "degraded": sum(1 for r in runs if r["status"] != "ok"),
    }
    return jsonify({"runs": runs, "totals": totals, "log_path": str(SETTINGS.run_log_path)})


@app.get("/api/run/<run_id>")
def api_run(run_id: str):
    for record in reversed(_load_runs()):
        if record.get("run_id") == run_id:
            record.pop("_line", None)
            return jsonify(record)
    return jsonify({"error": f"run {run_id} not found"}), 404


@app.get("/api/eval-results")
def api_eval_results():
    payload = _load_eval()
    if payload is None:
        return (
            jsonify(
                {
                    "error": "No eval results yet.",
                    "fix": "Run: python eval/run_eval.py",
                    "expected_path": str(SETTINGS.results_dir / "eval_results.json"),
                }
            ),
            404,
        )
    return jsonify(payload)


@app.get("/api/health")
def api_health():
    return jsonify(
        {
            "ok": True,
            "log_exists": SETTINGS.run_log_path.exists(),
            "eval_exists": (SETTINGS.results_dir / "eval_results.json").exists(),
            "model": SETTINGS.ollama_model,
            "worker_pool": SETTINGS.search_workers,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="app.py", description="Research pipeline dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"dashboard: http://{args.host}:{args.port}  (reading {SETTINGS.run_log_path})")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
