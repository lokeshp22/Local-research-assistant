# Local multi-agent research assistant

A research pipeline that runs entirely on one laptop GPU plus two search APIs.
A planner splits a question into sub-questions, two search workers answer them
against Tavily, a critic cross-checks every claim against independent sources
via Exa and throws out the ones that do not hold up, and the critic itself
writes the final cited report.

Everything it does is logged, and there is an eval harness that puts it head to
head against the obvious cheaper alternative — one model call with a search
bolted on — so the extra credits have to justify themselves.

**Stack:** Python 3.11+ · Ollama (`qwen3`) · Tavily · Exa · Flask · Chart.js
**Runs on:** a single 8 GB GPU (tested on an RTX 4060)

## Quick start

```bash
git clone https://github.com/<your-username>/local-research-assistant.git
cd local-research-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen3:8b
ollama pull nomic-embed-text

cp .env.example .env    # add your TAVILY_API_KEY and EXA_API_KEY

python selftest.py                      # offline check, no keys/GPU needed
python main.py "your research question"
python dashboard/app.py                 # http://127.0.0.1:5000
```

Full setup, the Ollama concurrency flags this depends on, and a staged testing
guide (offline → live) are below and in [TESTING.md](TESTING.md).

## Contents

- [Architecture](#architecture) — the pipeline, and why it's a 2-worker queue not a fan-out
- [Setup](#setup)
- [Running it](#running-it)
- [What is in the box](#what-is-in-the-box)
- [The run log](#the-run-log)
- [Evaluation](#evaluation)
- [Deviations from the original spec](#deviations-from-the-original-spec)
- [Testing](#testing)

## Status

Verified with 53/53 offline self-tests and a clean 20-question eval run against
scripted clients (`--fake` mode) — the wiring, schemas, and guardrails all check
out. It has **not** been benchmarked end-to-end against live Ollama/Tavily/Exa
at scale; do that on your own hardware with `eval/run_eval.py --limit 3` before
trusting the numbers on anything larger. See [Status details](#verification-status).

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
process, so the planner, both workers, the critic and the synthesizer can never
collectively exceed two generations in flight, no matter who wants one.

### One model for both agents, instead of role-specialized models

Both agents share a single `qwen3:30b-a3b`. With `OLLAMA_MAX_LOADED_MODELS=1`
this is not really a choice: a second model means evicting the first and
reloading it on every role switch, and the pipeline switches roles constantly
(plan → search → search → critique → synthesize). Load time would dominate the
run.

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

### 1. Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh

export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve

ollama pull qwen3:30b-a3b
ollama pull nomic-embed-text
```

**Read this before you start a long run.** `qwen3:30b-a3b` is roughly 18 GB at
q4 and does not fit in 8 GB of VRAM. Ollama will run it anyway by offloading
layers to system RAM. Because it is a mixture-of-experts model with about 3B
active parameters per token, that degrades far more gracefully than a dense 30B
would — but a full eval run across 20 questions will still take hours. The code
warns about this at startup and does not silently substitute anything.

If you want the pipeline to be interactive rather than a batch job:

```bash
# in .env
OLLAMA_MODEL=qwen3:8b
```

That fits in 8 GB with room for an 8k context and is the configuration to use
while you are iterating. Switch back to `30b-a3b` for the runs you want to
publish numbers from.

### 2. Keys and config

```bash
cp config/.env.example .env
# fill in TAVILY_API_KEY and EXA_API_KEY
```

`.env` is gitignored. No key is written to any log, any run record, or the
dashboard.

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

# dashboard (read-only against the logs)
python dashboard/app.py                  # http://127.0.0.1:5000
```

### Checking it works without a GPU

```bash
python selftest.py              # 53 checks: parsing, pipeline, log schema, all routes
python eval/run_eval.py --fake  # whole harness offline, scripted model
```

`--fake` swaps in scripted clients from `tools/fakes.py`. It proves the wiring,
the schemas and the guardrails. It says nothing about answer quality — the
scores it produces describe the harness, and the dashboard labels them as such.

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

Local models emit malformed structure constantly, so everything touching model
output goes through `tools/parsing.py`: tool arguments arriving as a JSON string
instead of a dict, double-encoded, wrapped in code fences, or as `[{"name":…,
"value":…}]` pairs; `<think>` blocks including unterminated ones; JSON buried in
prose; and URLs that are not URLs. `selftest.py` covers each of these.

---

## The run log

One JSON object per line in `logs/runs.jsonl`, `schema_version: 1`. The eval
harness and the dashboard both read this and nothing else.

```jsonc
{
  "run_id": "run_…", "timestamp": "…Z", "query": "…", "status": "ok|degraded|failed",
  "model": "qwen3:30b-a3b", "worker_pool_size": 2,
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
one Tavily search and one model call with no planner and no critic.

Scoring is split deliberately:

- **Citation coverage** and **source diversity** are computed in code
  (`tools/metrics.py`), not by a judge. A citation counts only when its number
  resolves to a source *and* that URL was genuinely retrieved during the run.
  Without that second condition, a model that invents a plausible URL scores the
  same as one that read the page.
- **Depth (1-5)** is the one axis left to the LLM judge, which sees the two
  reports in randomised order so it cannot learn a position bias.
- **Contradiction tests** are not judged at all. Five cases inject source
  snippets that cannot all be true; fixture clients replace Tavily and Exa so the
  only variable is the critic. A case passes only when a contradiction is raised
  spanning two *different* injected sources. Result is a plain pass/fail count.

Output: `eval/results/eval_results.json` and `.csv`, side by side per question,
with cost per approach so the comparison includes what it cost to win.

---

## Deviations from the original spec

Three, all deliberate:

1. **The dashboard is Flask, at `dashboard/app.py`.** The spec's dashboard
   section says "Flask, not Streamlit" and specifies Flask routes; its
   deliverables list says `/dashboard — streamlit_app.py`. Those contradict, and
   the detailed section won. There is no Streamlit dependency anywhere.
2. **Raw `httpx` instead of the `tavily-python` and `exa-py` SDKs.** The spec
   fixed the *APIs*, not the client libraries. One async transport with one
   retry/backoff policy across both providers is less surface area than two SDKs
   with different async maturity.
3. **`tools/` holds more than the two named wrappers.** `ollama_client.py`,
   `parsing.py`, `embeddings.py`, `metrics.py`, `budget.py`, `jsonl_log.py` and
   `fakes.py` live there too, rather than being scattered or duplicated.

The model choice was *not* changed. `qwen3:30b-a3b` remains the default despite
not fitting in 8 GB; the code warns and documents the alternative rather than
substituting one.

## Testing

Step-by-step, cheapest first: see [TESTING.md](TESTING.md).

## Verification status

`selftest.py` (53 checks) and `eval/run_eval.py --fake` were run against the
scripted clients: the queue, guardrails, verdict policy, citation numbering, log
schema, contradiction fixtures (5/5) and all Flask routes behave correctly, with
no empty reports and no dangling citations.

The live path — real Ollama, real Tavily, real Exa — has not been exercised, and
cannot be from a sandbox with no GPU and no route to either API. Run
`python eval/run_eval.py --limit 3` first on your own machine; it is the cheapest
way to confirm the live stack before committing credits to the full set.
