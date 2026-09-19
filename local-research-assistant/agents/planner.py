"""Planner — runs exactly once per query.

Turns a research question into 3-5 sub-questions via a tool call, validates the
result strictly, retries once on malformed output, then merges near-duplicates
using nomic-embed-text cosine similarity before anything is dispatched to the
search workers. Merging here rather than later is what stops two workers
burning Tavily credits on the same question in different words.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.budget import Budget
from tools.embeddings import merge_subquestions
from tools.ollama_client import OllamaError
from tools.parsing import as_str_list, extract_json

PLANNER_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_subquestions",
        "description": (
            "Emit the decomposition of the user's research question into "
            "independently searchable sub-questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sub_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Between 3 and 5 self-contained sub-questions. Each must be "
                        "answerable on its own by a web search, without reading the others."
                    ),
                }
            },
            "required": ["sub_questions"],
        },
    },
}

SYSTEM_PROMPT = """You are the planning stage of a research pipeline.

Break the user's research question into between {min_q} and {max_q} sub-questions.

Rules:
- Each sub-question must be self-contained. A search worker sees it alone, with no
  memory of the original question, so resolve every pronoun and implied subject.
- Cover genuinely different angles: definitions/mechanism, current state/data,
  competing views or criticism, and practical consequences. Do not restate one
  angle in different words.
- Each must be answerable from public web sources.
- Do not answer anything. Do not add commentary.

Call the emit_subquestions tool exactly once. Produce no other output."""

RETRY_HINT = """Your previous output was rejected: {error}

Call emit_subquestions once with a JSON object of the form:
{{"sub_questions": ["...", "...", "..."]}}
with between {min_q} and {max_q} strings. No prose, no markdown."""


class PlannerError(RuntimeError):
    """Fatal for the run — there is nothing to dispatch without a plan."""


@dataclass
class PlanResult:
    query: str
    sub_questions: list[str]
    raw_sub_questions: list[str] = field(default_factory=list)
    merges: list[dict] = field(default_factory=list)
    attempts: int = 1
    embedding_available: bool = True

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "raw_sub_questions": self.raw_sub_questions,
            "sub_questions": self.sub_questions,
            "merges": self.merges,
            "embedding_dedup_ran": self.embedding_available,
        }


async def plan(query: str, ollama, budget: Budget, settings) -> PlanResult:
    system = SYSTEM_PROMPT.format(min_q=settings.min_subquestions, max_q=settings.max_subquestions)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query.strip()},
    ]

    raw: list[str] = []
    last_error = ""
    attempts = 0

    for attempt in range(2):  # one initial attempt, one retry — then fail
        attempts = attempt + 1
        try:
            message = await ollama.chat(
                messages, tools=[PLANNER_TOOL], timeout=settings.call_timeout_s, retries=0
            )
        except OllamaError as exc:
            last_error = str(exc)
            budget.record_failure("planner", "ollama", last_error)
            messages.append(
                {
                    "role": "user",
                    "content": RETRY_HINT.format(
                        error=last_error,
                        min_q=settings.min_subquestions,
                        max_q=settings.max_subquestions,
                    ),
                }
            )
            continue

        candidate = _harvest(message)
        ok, error = _validate(candidate, settings)
        if ok:
            raw = candidate
            break

        last_error = error
        budget.record_failure("planner", "validation", error, attempt=attempts)
        messages.append({"role": "assistant", "content": message.get("content") or ""})
        messages.append(
            {
                "role": "user",
                "content": RETRY_HINT.format(
                    error=error,
                    min_q=settings.min_subquestions,
                    max_q=settings.max_subquestions,
                ),
            }
        )

    if not raw:
        raise PlannerError(f"Planner produced no valid sub-question list after {attempts} attempts: {last_error}")

    raw = raw[: settings.max_subquestions]

    # Embedding dedup. A failure here must not kill the run — we just skip the
    # merge and note that it did not happen.
    merged, notes, embedded = raw, [], True
    try:
        vectors = await ollama.embed(raw)
        if len(vectors) == len(raw):
            merged, notes = merge_subquestions(raw, vectors, settings.dedup_threshold)
        else:
            embedded = False
            budget.record_failure(
                "planner", "embeddings", f"expected {len(raw)} vectors, got {len(vectors)}"
            )
    except OllamaError as exc:
        embedded = False
        budget.record_failure("planner", "embeddings", str(exc))

    return PlanResult(
        query=query,
        sub_questions=merged,
        raw_sub_questions=raw,
        merges=notes,
        attempts=attempts,
        embedding_available=embedded,
    )


def _harvest(message: dict) -> list[str]:
    """Prefer the tool call; fall back to JSON in the content, because local
    models regularly describe the call instead of making it."""
    for call in message.get("tool_calls") or []:
        if call["name"] != "emit_subquestions":
            continue
        items = as_str_list(call["arguments"].get("sub_questions"))
        if items:
            return items
        items = as_str_list(call["arguments"])
        if items:
            return items

    parsed = extract_json(message.get("content") or "")
    if isinstance(parsed, dict):
        for key in ("sub_questions", "subquestions", "questions"):
            items = as_str_list(parsed.get(key))
            if items:
                return items
    if isinstance(parsed, list):
        return as_str_list(parsed)
    return []


def _validate(items: list[str], settings) -> tuple[bool, str]:
    if not items:
        return False, "no sub_questions array was produced"
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = item.strip().lower().rstrip("?")
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    items[:] = deduped

    if len(items) < settings.min_subquestions:
        return False, f"got {len(items)} distinct sub-questions, need at least {settings.min_subquestions}"
    if any(len(item.strip()) < 12 for item in items):
        return False, "one or more sub-questions were too short to be searchable"
    return True, ""
