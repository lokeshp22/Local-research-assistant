#!/usr/bin/env python3
"""CLI entrypoint.

    python main.py "How is grid-scale battery storage actually being financed?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import ConfigError, load_settings  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from tools.budget import Budget  # noqa: E402
from tools.exa_client import ExaClient  # noqa: E402
from tools.ollama_client import OllamaClient, OllamaError  # noqa: E402
from tools.tavily_client import TavilyClient  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py", description="Two-agent local research assistant."
    )
    parser.add_argument("query", help="The research question, in quotes.")
    parser.add_argument("--model", help="Override OLLAMA_MODEL for this run.")
    parser.add_argument("--no-log", action="store_true", help="Do not append to the JSONL run log.")
    parser.add_argument("--json", action="store_true", help="Print the full run record as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Print the report only.")
    return parser


async def _run(args) -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.model:
        settings = type(settings)(**{**settings.__dict__, "ollama_model": args.model})

    try:
        settings.require_search_keys()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        settings.warn_if_oversized()

    budget = Budget()
    ollama = OllamaClient(settings, budget)
    tavily = TavilyClient(settings, budget)
    exa = ExaClient(settings, budget)

    try:
        try:
            await ollama.preflight()
        except OllamaError as exc:
            print(f"ollama error: {exc}", file=sys.stderr)
            return 3

        if not args.quiet:
            print(
                f"[run] model={settings.ollama_model} workers={settings.search_workers} "
                f"timeout={settings.call_timeout_s}s max_turns={settings.max_turns}",
                file=sys.stderr,
            )

        orchestrator = Orchestrator(settings, ollama=ollama, tavily=tavily, exa=exa, budget=budget)
        record = await orchestrator.run(args.query, log=not args.no_log)
    finally:
        await asyncio.gather(
            ollama.aclose(), tavily.aclose(), exa.aclose(), return_exceptions=True
        )

    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0 if record["status"] != "failed" else 1

    _print_report(record, quiet=args.quiet)
    return 0 if record["status"] != "failed" else 1


def _print_report(record: dict, *, quiet: bool) -> None:
    print()
    print(record["final_report"] or "(no report produced)")
    if record["sources"]:
        print("\n## Sources\n")
        for source in record["sources"]:
            print(f"[{source['index']}] {source['url']}")
    if quiet:
        return

    counts = record["verdict_counts"]
    metrics = record.get("metrics", {})
    print(
        f"\n---\nrun_id={record['run_id']} status={record['status']}\n"
        f"claims: {counts.get('kept',0)} kept / {counts.get('flagged',0)} flagged / "
        f"{counts.get('rejected',0)} rejected · "
        f"{metrics.get('source_diversity',0)} unique domains",
        file=sys.stderr,
    )
    budget = record["budget"]
    print(
        f"budget: {budget['wall_clock_s']}s wall · "
        f"Tavily {budget['tavily']['credits_estimated']} cr · "
        f"Exa {budget['exa']['credits_estimated']} cr · "
        f"Ollama {budget['ollama']['chat_calls']} chat + {budget['ollama']['embed_calls']} embed",
        file=sys.stderr,
    )
    if record["failures"]:
        print(f"failures ({len(record['failures'])}):", file=sys.stderr)
        for failure in record["failures"][:10]:
            print(f"  - [{failure['stage']}/{failure['component']}] {failure['detail']}", file=sys.stderr)


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
