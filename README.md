# 🔬 Local Multi-Agent Research Assistant

*Four AI agents argue about your research question until only the true claims survive.*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Runs on 8GB GPU](https://img.shields.io/badge/GPU-8GB%20VRAM-orange)](#setup)

---

## What is this thing?

You ask a research question. Behind the scenes:

1. A **planner** breaks it into 3–5 sharper sub-questions.
2. Two **search workers** race through them in parallel, each pulling real sources off the web via Tavily.
3. A **critic** — the skeptic of the operation — cross-checks every single claim against independent sources via Exa, and throws out anything that doesn't hold up. No mercy. No vibes-based fact-checking.
4. That same critic writes the final report, citing only what survived.

Every step is logged. There's a dashboard to watch it happen. And there's an eval harness that pits the whole four-stage production against the boring, cheap alternative — one model, one search, no fact-checking — because if the fancy pipeline can't beat that, what's the point?

It's designed to run **entirely on a laptop GPU with 8 GB of VRAM.** No cloud LLM, no API bill for the "thinking" part — just Ollama, running locally, doing the actual reasoning while Tavily and Exa handle the web.

**Stack:** Python 3.11+ · Ollama (`qwen3`) · Tavily · Exa · Flask · Chart.js
**Runs on:** a single 8 GB GPU (tested on an RTX 4060)

---

## Quick start

```bash
git clone https://github.com/<your-username>/local-research-assistant.git
cd local-research-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ollama needs these BEFORE you start the server — see "Setup" for why
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
ollama serve

ollama pull qwen3:8b
ollama pull nomic-embed-text

cp .env.example .env    # add your TAVILY_API_KEY and EXA_API_KEY

python selftest.py                      # 53 checks, no GPU or keys required
python main.py "your research question"
python dashboard/app.py                 # http://127.0.0.1:5000
```

If your first live query hangs for a suspiciously long time, it's not broken — see [Battle scars](#battle-scars-things-that-actually-went-wrong) before you panic.

## Contents

- [Architecture](#architecture) — the pipeline, and why it's a 2-worker queue and not a free-for-all
- [Setup](#setup)
- [Running it](#running-it)
- [What is in the box](#what-is-in-the-box)
- [The run log](#the-run-log)
- [Evaluation](#evaluation)
- [Battle scars: things that actually went wrong](#battle-scars-things-that-actually-went-wrong)
- [Deviations from the original spec](#deviations-from-the-original-spec)
- [Testing](#testing)

## Status

Verified with 53/53 offline self-tests and a clean 20-question eval run against
scripted clients (`--fake` mode) — the wiring, schemas, and guardrails all check
out. It has **not** been benchmarked end-to-end against live Ollama/Tavily/Exa
at scale; do that on your own hardware with `eval/run_eval.py --limit 3` before
trusting the numbers on anything larger. See [Verification status](#verification-status).

## License

[MIT](LICENSE) — swap it for whatever you prefer before publishing if MIT
isn't the right fit.

---

## Architecture

```
  "How is grid-scale storage financed?"
                  │
                  ▼
        ┌───────────────────┐
        │      PLANNER      │  1 model call (+1 retry on bad JSON)
        │  emit_subquestions│  strict JSON validation, cap 5
        └─────────┬─────────┘
                  │  nomic-embed-text: merge pairs with cosine > 0.90
                  ▼
        ┌───────────────────┐
        │   asyncio.Queue   │   3-5 sub-questions
        └──┬─────────────┬──┘
           │             │            ← exactly 2 workers. Not gather().
     ┌─────▼─────┐ ┌─────▼─────┐         See "The concurrency constraint".
     │ SEARCH  1 │ │ SEARCH  2 │  per sub-question:
     │           │ │           │    tavily_search → tavily_extract(3-4)
     │ ≤5 turns  │ │ ≤5 turns  │    → emit_claims(text, url, confidence)
     │ 30s/call  │ │ 30s/call  │  every URL checked against what was retrieved
     └─────┬─────┘ └─────┬─────┘
           └──────┬──────┘
                  ▼  queue drains, then:
        ┌───────────────────┐
        │      CRITIC       │  per claim: Exa search (own domain excluded)
        │                   │            + 1 verdict call
        │  kept / flagged   │  across claims: contradiction checks on pairs
        │  / rejected       │            that are about the same thing
        └─────────┬─────────┘
                  │  same agent, final call — not a fifth model
                  ▼
        ┌───────────────────┐
        │    SYNTHESIS      │  kept claims only, inline [1][2],
        │                   │  citation numbers assigned before generation
        └─────────┬─────────┘
                  ▼
     logs/runs.jsonl  ──→  eval harness  ──→  Flask dashboard
```

Nobody's claim gets into the final report on the strength of confidence alone.
A claim survives only if: the URL it cites was actually retrieved (checked in
code, not by asking the model nicely), an independent source agrees or at
least doesn't contradict it, and the source itself looks like it was written
by someone real. Fail any of those and you get flagged or rejected, with a
one-line reason logged so you can see exactly why.

### The concurrency constraint, and why this is 2-way pipelined

The design target is 8 GB of VRAM on a single GPU (an RTX 4060 laptop), with
Ollama launched as `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_MAX_LOADED_MODELS=1`.

Ollama does not do continuous batching. A third concurrent request does not
join an in-flight forward pass — it waits in Ollama's own queue while still
wanting KV-cache space. So `asyncio.gather` over an unbounded list of
sub-questions would produce the worst of both worlds: no extra throughput, more
memory pressure, and worse tail latency on the requests already running.

The fan-out is therefore bounded at the source. Sub-questions go onto an
`asyncio.Queue` and exactly `search_workers` (= `OLLAMA_NUM_PARALLEL` = 2)
workers pull from it. A worker that finishes early picks up the next
sub-question immediately — there is no barrier between sub-questions, only the
drain of the queue before the critic starts. That is the difference between
*pipelined* and *batched*.

One `asyncio.Semaphore(2)` inside `OllamaClient` is shared by the whole
process, so the planner, both workers, the critic, and the synthesizer can
never collectively exceed two generations in flight, no matter who wants one.

### One model for both agents, instead of role-specialized models

Both agents share a single model. With `OLLAMA_MAX_LOADED_MODELS=1` this is
not really a choice: a second model means evicting the first and reloading it
on every role switch, and the pipeline switches roles constantly (plan →
search → search → critique → synthesize). Load time would dominate the run.

What that costs: the critic inherits the searcher's blind spots. A model is a
soft grader of its own output, and asking it to fact-check text shaped by the
same weights is weaker than a genuinely independent reviewer. The design
compensates by keeping as much of the verdict as possible *outside* the model:

- the cited URL is checked against what the worker actually retrieved — code, not judgement;
- Exa cross-checks exclude the claim's own domain, so corroboration cannot be the same page again;
- the kept/flagged/rejected policy is a deterministic function (`_apply_verdict`) of the model's three narrow judgements, not a free-text verdict;
- contradiction candidates are selected by similarity, so the model answers "do these two conflict?" rather than "find all the problems".

If you have the VRAM, running the critic on a different model family is the
single highest-value upgrade here. Point `OLLAMA_MODEL` at the searcher's model
and add a separate client for the critic; the interfaces already allow it.

---

## Setup

### 1. Ollama — and the three environment variables that actually matter

```bash
curl -fsSL https://ollama.com/install.sh | sh

export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=30m
ollama serve

ollama pull qwen3:8b
ollama pull nomic-embed-text
```

That third variable, `OLLAMA_KEEP_ALIVE`, is easy to skip and will quietly
wreck your afternoon if you do. Without it, Ollama can unload the model
between calls and reload it from disk on the next one — and a reload can eat
40+ seconds *before generation even starts.* With ~20-40 model calls in a
single pipeline run, that's not a slowdown, it's a different runtime
complexity class. Set it and forget it.

**On bigger models:** `qwen3:30b-a3b` (the model this project was originally
speced against) is roughly 18 GB at q4 and does not fit in 8 GB of VRAM.
Ollama will run it anyway by offloading layers to system RAM. Because it's a
mixture-of-experts model with only ~3B active parameters per token, it
degrades far more gracefully than a dense 30B would — but a full eval run
across 20 questions will still take hours. The code warns about this at
startup and never silently substitutes a different model for you.

For anything interactive, start with `qwen3:8b` — it fits fully in 8 GB with
room to spare for an 8k context window, and it's the config to develop against.
Switch to `30b-a3b` only for runs you actually want to publish numbers from.

### 2. Keys and config

```bash
cp config/.env.example .env
# fill in TAVILY_API_KEY and EXA_API_KEY
```

`.env` is gitignored. No key is written to any log, any run record, or the
dashboard — go check `tools/budget.py` if you don't believe it; it tracks
*credit counts*, never the keys themselves.

### 3. Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # httpx, python-dotenv, numpy, flask
```

---

## Running it

```bash
# one question, prints the report
python main.py "How is grid-scale battery storage actually financed?"

# full comparison against the baseline, plus the contradiction tests
python eval/run_eval.py
python eval/run_eval.py --limit 3        # smoke test first — it is much cheaper

# dashboard (read-only against the logs, never writes to them)
python dashboard/app.py                  # http://127.0.0.1:5000
```

### Checking it works without spending a credit or owning a GPU

```bash
python selftest.py              # 53 checks: parsing, pipeline, log schema, all routes
python eval/run_eval.py --fake  # the whole harness, offline, scripted model
```

`--fake` swaps in scripted clients from `tools/fakes.py` in place of Ollama,
Tavily, and Exa. It proves the wiring, the schemas, and the guardrails all
work. It says **nothing** about answer quality — the scores it produces
describe the harness, not the model, and the dashboard labels them "offline"
so you never mistake a plumbing test for a real result.

---

## What is in the box

```
config/       .env.example, settings.py          all tunables in one place
tools/        tavily_client.py, exa_client.py    thin async wrappers, retry + backoff
              ollama_client.py                   shared 2-slot semaphore, tool calls, embeddings
              parsing.py                         defensive parsing (see below)
              embeddings.py, metrics.py          similarity, citation scoring
              budget.py, jsonl_log.py            credit accounting, append-only log
              fakes.py                           fixtures for contradiction tests + offline clients
agents/       planner.py, search_agent.py, critic_agent.py
orchestrator.py                                  the 2-worker queue
eval/         eval_questions.json                20 held-out questions
              contradiction_test_cases.json      5 seeded conflicts
              run_eval.py, results/
dashboard/    app.py, templates/index.html       Flask + vanilla JS + Chart.js (CDN)
logs/         runs.jsonl
main.py, selftest.py
```

### Guardrails

| Guardrail | Where |
|---|---|
| 30s hard timeout on every model call and every HTTP call | `asyncio.wait_for` in both clients and both agents |
| Max 5 tool turns per sub-question | `search_agent.run_sub_question` |
| Max 5 sub-questions, min 3, one retry on bad JSON | `planner.plan` |
| Tavily/Exa failure degrades instead of crashing | failures become tool *results* the model can route around |
| A crashed worker does not take down the run | `Orchestrator._drain_queue` |
| Per-run credits, wall clock, model calls | `tools/budget.py`, logged per run |
| Contradiction-check ceiling | `MAX_CONTRADICTION_CHECKS` |

### Defensive parsing

Local models emit malformed structure *constantly.* Not occasionally — as a
matter of routine. So everything touching model output goes through
`tools/parsing.py`, which handles: tool arguments arriving as a JSON string
instead of a dict, double-encoded strings, arguments wrapped in markdown code
fences, or shaped as `[{"name":…, "value":…}]` pairs instead of a plain
object; `<think>` reasoning blocks (including ones that never got a closing
tag); JSON buried in the middle of a paragraph of prose; and URLs that, on
closer inspection, are not actually URLs. `selftest.py` exercises every one of
these on purpose — they're not hypothetical, they all happened during
development.

---

## The run log

One JSON object per line in `logs/runs.jsonl`, `schema_version: 1`. The eval
harness and the dashboard both read this and nothing else.

```jsonc
{
  "run_id": "run_…", "timestamp": "…Z", "query": "…", "status": "ok|degraded|failed",
  "model": "qwen3:8b", "worker_pool_size": 2,
  "planner": { "attempts": 1, "raw_sub_questions": [...], "sub_questions": [...],
               "merges": [{ "kept": "…", "merged_away": ["…"], "max_similarity": 0.93 }] },
  "sub_questions": [{ "sub_question": "…", "status": "ok", "turns": 3,
                      "claim_count": 4, "sources_seen": [...], "failures": [] }],
  "claims": [{ "id": "c_…", "text": "…", "source_url": "…", "url_verified": true,
               "confidence": "high", "verdict": "kept|flagged|rejected", "reason": "…",
               "support": "…", "corroboration": "…", "source_quality": "…",
               "cross_check_sources": [...] }],
  "contradictions": [{ "claim_ids": [...], "explanation": "…", "similarity": 0.81 }],
  "final_report": "…", "sources": [{ "index": 1, "url": "…" }],
  "timing_s": { "planner": …, "search": …, "critic": …, "synthesis": …, "total": … },
  "budget": { "tavily": { "searches": 8, "credits_estimated": 8.0 },
              "exa": { "searches": 12, "credits_estimated": 19.2 },
              "ollama": { "chat_calls": 38, "embed_calls": 2 } },
  "failures": [{ "stage": "search", "component": "tavily_search", "detail": "…" }],
  "metrics": { "citation_coverage": …, "source_diversity": …, "claim_url_verification": … }
}
```

**Credit figures are an accounting model, not a provider readout.** The rates
live in `tools/budget.py` and come from the published pricing at build time.
Correct them there against your real usage dashboard; nothing else needs to
change.

The critic is where the money goes: one Exa search plus one model call per
claim. Two optimisations are already in: claims whose cited URL was never
retrieved are rejected without a cross-check (same verdict, zero cost), and
contradiction checking is capped. If a run is still too expensive, lower
`MAX_CONTRADICTION_CHECKS` and `EXA_NUM_RESULTS` first.

---

## Evaluation

Each question runs twice: through the full pipeline, and through a baseline of
one Tavily search and one model call with no planner and no critic. That
baseline exists specifically to be beaten — if four agents and two APIs can't
outperform "search once, ask once," the extra complexity isn't earning its
keep.

Scoring is split deliberately, because letting a model score its own citations
is exactly the kind of thing this whole project exists to prevent:

- **Citation coverage** and **source diversity** are computed in code
  (`tools/metrics.py`), never by a judge. A citation counts only when its
  number resolves to a source *and* that URL was genuinely retrieved during
  the run. Without that second condition, a model that invents a plausible-
  looking URL scores identically to one that actually read the page.
- **Depth (1-5)** is the one axis left to an LLM judge, and it sees the two
  reports in randomised order so it can't learn a position bias.
- **Contradiction tests** aren't judged by a model at all. Five cases inject
  source snippets that cannot both be true; fixture clients stand in for
  Tavily and Exa so the critic is the only variable. A case passes only when a
  contradiction is raised spanning two *different* injected sources. The
  result is a plain, boring, trustworthy pass/fail count.

Output: `eval/results/eval_results.json` and `.csv`, side by side per
question, with cost per approach — so the comparison includes what it cost to
win, not just whether it won.

---

## Battle scars: things that actually went wrong

Nobody writes these in a README, which is a shame, because they're usually the
most useful part. In order of "how long it took to figure out":

**The contradiction detector was silently blind to the exact thing it exists
to catch.** Two claims — "capacity rose 14%" and "capacity fell 6%" — are
about as clean a contradiction as English allows. Embedding cosine similarity
alone missed it, because a statement and its negation often land *far apart*
in embedding space; the model represents "meaning," and "rose" versus "fell"
is a big swing in meaning even though the sentences are about the same fact.
Fixed by adding a plain lexical-overlap check alongside the embedding
similarity — if two claims share most of their content words, they're worth
comparing regardless of what the vectors say.

**URL normalization accepted garbage as a valid source.** `normalize_url`
happily turned `HTTP://WWW.Example.com` into a mangled double-prefixed mess,
and worse, accepted strings like `::::` as if they were legitimate hosts. Now
every candidate URL is checked against an actual hostname pattern before it's
allowed anywhere near a claim.

**Citation coverage meant two different things depending on which file you
were reading.** The live orchestrator computed it one way, the eval harness
computed it another way, and the numbers on the dashboard quietly disagreed
with the numbers in the CSV. Now both read from one shared
`tools/metrics.py` — there is exactly one definition of "citation coverage"
in this codebase, and it's testable in isolation.

**The most expensive bug wasn't a bug — it was Ollama unloading the model
between calls.** A single `"hi"` was taking 45+ seconds to answer. `nvidia-smi`
showed the GPU basically idle; `ollama ps` confirmed it: the model wasn't
resident, it was reloading from disk before every single generation. Twenty
Ollama calls later — planner, two workers, per-claim critic checks,
contradiction checks, synthesis — that 45-second tax had compounded into a
run that took over five minutes instead of under one. The fix was one
environment variable: `OLLAMA_KEEP_ALIVE=30m`. It's now called out explicitly
in Setup, in bold, because it will happen to you too if you skip it.

---

## Deviations from the original spec

Three, all deliberate:

1. **The dashboard is Flask, at `dashboard/app.py`.** The spec's dashboard
   section says "Flask, not Streamlit" and specifies Flask routes; its
   deliverables list says `/dashboard — streamlit_app.py`. Those contradict,
   and the detailed section won. There is no Streamlit dependency anywhere.
2. **Raw `httpx` instead of the `tavily-python` and `exa-py` SDKs.** The spec
   fixed the *APIs*, not the client libraries. One async transport with one
   retry/backoff policy across both providers is less surface area than two
   SDKs with different async maturity.
3. **`tools/` holds more than the two named wrappers.** `ollama_client.py`,
   `parsing.py`, `embeddings.py`, `metrics.py`, `budget.py`, `jsonl_log.py`,
   and `fakes.py` live there too, rather than being scattered around the repo
   or duplicated.

The model choice was *not* changed. `qwen3:30b-a3b` remains supported despite
not fitting in 8 GB on its own; the code warns and documents the lighter
alternative rather than silently substituting one for you.

## Testing

Step-by-step, cheapest first: see [TESTING.md](TESTING.md). It walks through
an offline plumbing check, sanity-testing Ollama on its own, one live query,
and then a small eval run before you commit to the full 20-question set — in
that order, because every problem worth catching shows up early and cheap.

## Verification status

`selftest.py` (53 checks) and `eval/run_eval.py --fake` were run against
scripted clients: the queue, guardrails, verdict policy, citation numbering,
log schema, contradiction fixtures (5/5), and every Flask route all behave
correctly, with zero empty reports and zero dangling citations.

The live path — real Ollama, real Tavily, real Exa — has been exercised
manually during development (see Battle scars above) but not benchmarked at
scale. Run `python eval/run_eval.py --limit 3` first on your own machine; it's
the cheapest way to confirm the live stack behaves before committing real
credits to the full set.
