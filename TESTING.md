# Testing this yourself

Four stages, cheapest first. Don't skip to stage 4 — a full eval run on a 30B
model with a 4060 is an hours-long, credit-spending job, and every problem worth
catching shows up earlier.

---

## Stage 0 — offline, no GPU, no API keys (2 minutes)

This proves the wiring, the guardrails and the schemas without touching Ollama
or spending a credit. Do this immediately after unzipping.

```bash
cd local-research-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python selftest.py
```

Expected last line: `53 passed, 0 failed`.

It covers defensive parsing (tool args as JSON strings, double-encoded,
fenced, `<think>` blocks, malformed URLs), the planner's dedup/merge, the
2-worker queue, tool-turn caps, the critic's verdict policy, citation
numbering, graceful degradation when both providers throw, all five seeded
contradiction cases, the JSONL schema, and every Flask route including the
check that the dashboard never writes to the log.

Then run the whole eval harness offline:

```bash
python eval/run_eval.py --fake
```

Expected: `20` questions, `contradiction tests: 5/5 passed`, `empty reports 0`
for both approaches.

**What this does not tell you.** `--fake` swaps in a scripted model from
`tools/fakes.py`. The depth scores it prints describe the harness, not the
model. Runs produced this way are tagged `mode: "offline"` in the log and the
dashboard labels them — don't read them as a measurement.

### See the dashboard with data in it

The zip ships with the offline run above already in `logs/runs.jsonl`, so the
dashboard has something to render on first launch:

```bash
python dashboard/app.py
# http://127.0.0.1:5000
```

Check: the claim gate at the top shows kept/flagged/rejected proportions, both
charts render, the contradiction summary reads `5/5 passed`, and clicking a row
expands sub-questions, per-claim verdicts with reasons, and the report.

When you want a clean slate:

```bash
rm -f logs/runs.jsonl eval/results/eval_results.*
```

---

## Stage 1 — Ollama on its own (10 minutes, plus pull time)

Before involving this project at all, confirm the model actually generates and
tool-calls on your machine.

```bash
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve          # leave running in its own terminal

ollama pull nomic-embed-text
ollama pull qwen3:8b            # start here, not 30b-a3b
```

Sanity-check the server and both models:

```bash
curl -s localhost:11434/api/tags | grep -o '"name":"[^"]*"'

curl -s localhost:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":["hello"]}' | head -c 120
```

The embed call must return an `embeddings` array. If it 404s you're on an older
Ollama — the client falls back to `/api/embeddings` automatically, so that's
fine, but note it.

Now the part that actually matters, tool calling:

```bash
curl -s localhost:11434/api/chat -d '{
  "model":"qwen3:8b","stream":false,
  "messages":[{"role":"user","content":"List three sub-questions about solar financing."}],
  "tools":[{"type":"function","function":{
    "name":"emit_subquestions",
    "parameters":{"type":"object","properties":{
      "sub_questions":{"type":"array","items":{"type":"string"}}},
    "required":["sub_questions"]}}}]
}' | python3 -m json.tool | head -30
```

You want a `tool_calls` entry in the response. If the model answers in prose
instead, the planner will still recover (it parses JSON out of the content as a
fallback), but expect more retries.

**On `qwen3:30b-a3b`:** it's ~18 GB at q4 and your 4060 has 8 GB, so Ollama
offloads most of it to system RAM. It's MoE with ~3B active params so it stays
usable, but it is slow. Test everything on `qwen3:8b` first and only switch for
the runs you want to quote numbers from. Watch `nvidia-smi` and `ollama ps`
during a run — if `ollama ps` shows a large CPU percentage, that's the offload.

---

## Stage 2 — one live query (5 minutes, ~10 credits)

```bash
cp config/.env.example .env
```

Edit `.env`: paste your **newly rotated** Tavily and Exa keys, and set
`OLLAMA_MODEL=qwen3:8b` for now.

```bash
python main.py "What changed in EU battery storage subsidy rules in 2025?"
```

The run prints the report and sources to stdout, and a summary line to stderr.

**What to check, in order:**

1. It finished at all. Status `ok` or `degraded` is fine; `failed` is not.
2. The planner produced 3–5 sub-questions. Check the log:
   ```bash
   python3 -c "import json;r=json.loads(open('logs/runs.jsonl').readlines()[-1]);print(json.dumps(r['planner'],indent=2))"
   ```
3. **Open two of the cited URLs and read them.** This is the one check nothing
   automated can do for you. The pipeline guarantees a cited URL was genuinely
   retrieved; it cannot guarantee the page says what the claim says.
4. Look at what the critic threw out and why:
   ```bash
   python3 -c "
   import json;r=json.loads(open('logs/runs.jsonl').readlines()[-1])
   for c in r['claims']:
       if c['verdict']!='kept': print(c['verdict'].upper(),'|',c['reason'],'|',c['text'][:90])"
   ```
   Rejections reading *cited URL was never returned by the worker's own
   searches* mean the model is inventing sources — that's the check working.
5. Check the cost:
   ```bash
   python3 -c "import json;r=json.loads(open('logs/runs.jsonl').readlines()[-1]);print(json.dumps(r['budget'],indent=2))"
   ```

**Red flags and what they mean:**

| What you see | Likely cause | Fix |
|---|---|---|
| Every claim rejected for unverified URL | Model isn't copying URLs verbatim | Try a larger model, or raise `OLLAMA_NUM_CTX` so results stay in context |
| Sub-questions look like restatements of each other | Planner isn't decomposing | Check `planner.merges` — if dedup fired, it's working; if not, lower `DEDUP_THRESHOLD` to 0.85 |
| `turns: 5` on every sub-question | Agent never calls `emit_claims` | Model is weak at tool calling; the final harvest call catches this but quality drops |
| Lots of Ollama timeouts | 30s is too tight for a CPU-offloaded 30B | Raise `CALL_TIMEOUT_S` to 90 |
| Zero contradictions ever | Normal on most questions | Verify the mechanism with the seeded tests instead |

---

## Stage 3 — small live eval (30–60 min on qwen3:8b, ~60 credits)

```bash
python eval/run_eval.py --limit 3
```

This runs 3 questions through both approaches plus 3 contradiction cases, and
writes `eval/results/eval_results.json` and `.csv`.

**What good looks like:**

- `empty reports: 0` for both approaches.
- Multi-agent `citation coverage` at or near `1.0`. Below ~0.9 means the
  synthesizer is emitting citation numbers that don't resolve — a real bug worth
  investigating, not noise.
- Multi-agent `source diversity` above baseline. If it isn't, the planner's
  sub-questions are too similar and the workers are finding the same pages.
- `contradiction tests: 3/3 passed`. This is the one number that must be
  perfect — it's a controlled test with a known answer. A failure here means the
  critic is broken, not that the question was hard.
- Depth: expect multi-agent to edge baseline. A small gap is normal and honest;
  a large one on a 3-question sample is noise, not evidence.

The credit rows tell you what winning cost. Tavily ~8 vs 1 per question, Exa
~19 vs 0, Ollama calls ~38 vs 1. That ratio is the real question this project
exists to answer.

---

## Stage 4 — the full run

```bash
python eval/run_eval.py          # 20 questions + 5 contradiction cases
python dashboard/app.py
```

Budget roughly: 20 questions × (~8 Tavily + ~19 Exa credits) ≈ 160 Tavily and
380 Exa credits, plus ~800 Ollama calls. On `qwen3:30b-a3b` with CPU offload
that is an overnight job. Run it with `nohup` or in `tmux`.

If it's too expensive, the three knobs in `.env`, in order of impact:

```
MAX_CONTRADICTION_CHECKS=6    # fewer claim-pair comparisons
EXA_NUM_RESULTS=3             # cheaper cross-checks
MAX_SUBQUESTIONS=4            # fewer workers' worth of searching
```

---

## Testing the failure paths deliberately

Worth doing once — graceful degradation is only real if you've seen it.

**Kill a provider mid-run:** put a bad Tavily key in `.env` and run
`python main.py "anything"`. The run must complete with `status: degraded`,
failures recorded in the log, and a report built from whatever Exa and the
remaining sources gave. It must not traceback.

**Kill Ollama mid-run:** start a query, then `pkill ollama` a few seconds in.
Same expectation: recorded failures, no traceback.

**Confirm the dashboard is read-only:** note the file size, click around every
view, check it again.

```bash
stat -c%s logs/runs.jsonl   # before and after
```

`selftest.py` asserts this too, but seeing it is worth more.
