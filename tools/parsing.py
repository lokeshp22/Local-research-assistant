"""Defensive parsing.

Local models emit malformed structure constantly: tool arguments arrive as JSON
strings instead of objects, double-encoded, wrapped in code fences, or with a
<think> block glued to the front. Everything that touches model output goes
through here so the failure modes are handled in one place.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|javascript|js)?\s*(.*?)```", re.DOTALL)
_HOST_RE = re.compile(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+")


def strip_think(text: str) -> str:
    """Remove qwen3 reasoning blocks, including an unterminated trailing one."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def coerce_tool_args(raw: Any) -> dict:
    """Turn whatever came back in `tool_call.function.arguments` into a dict.

    Handles: dict, JSON string, double-encoded JSON string, fenced JSON,
    and None. Returns {} rather than raising — the caller validates fields.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)):
        # Some runtimes hand back [{"name": ..., "value": ...}] pairs.
        out: dict = {}
        for item in raw:
            if isinstance(item, dict) and "name" in item:
                out[str(item["name"])] = item.get("value")
        return out
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        for candidate in _json_candidates(raw):
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            # Double-encoded: json.loads gave us another string.
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except (json.JSONDecodeError, ValueError):
                    continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"_list": parsed}
    return {}


def extract_json(text: str) -> Any:
    """Pull the first parseable JSON value out of free-form model text."""
    if not text:
        return None
    cleaned = strip_think(text)
    for candidate in _json_candidates(cleaned):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _json_candidates(text: str):
    text = strip_think(text).strip()
    if not text:
        return
    yield text
    for fenced in _FENCE_RE.findall(text):
        yield fenced.strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        block = _balanced_block(text, opener, closer)
        if block:
            yield block


def _balanced_block(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def as_str_list(value: Any) -> list[str]:
    """Coerce a 'list of strings' field that may be a string, a comma list, or
    a list of dicts with a text-ish key."""
    if value is None:
        return []
    if isinstance(value, str):
        parsed = extract_json(value)
        if isinstance(parsed, list):
            return as_str_list(parsed)
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        for key in ("items", "list", "values", "sub_questions", "subquestions"):
            if key in value:
                return as_str_list(value[key])
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("question", "sub_question", "subquestion", "text", "q", "value"):
                    if isinstance(item.get(key), str) and item[key].strip():
                        out.append(item[key].strip())
                        break
        return out
    return []


def normalize_url(url: Any) -> str:
    if not isinstance(url, str):
        return ""
    url = url.strip().strip("<>").rstrip(".,;)")
    if not url:
        return ""
    if not url.lower().startswith(("http://", "https://")):
        if "://" in url:  # some other scheme (ftp:, javascript:, data:) — not a web source
            return ""
        url = "https://" + url.lstrip("/")
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().strip()
    if not netloc:
        return ""
    host = netloc.split("@")[-1].split(":")[0]
    # A usable web host has a dot and no structural junk. This is what keeps a
    # hallucinated "source" like "::::" or "source 3" out of the claim list.
    if not _HOST_RE.fullmatch(host) and host != "localhost":
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme.lower()}://{netloc}{path}" + (f"?{parsed.query}" if parsed.query else "")


def domain_of(url: str) -> str:
    if not url:
        return ""
    netloc = urlparse(url if "://" in url else "https://" + url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.split(":")[0]


def truncate(text: Any, limit: int) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
